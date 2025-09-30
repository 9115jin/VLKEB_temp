from .BaseTrainer import *
import json
import logging
import os
import shutil
import tempfile
import time
import numpy as np

import torch
from .losses import kl_loc_loss
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from .utils import (
    EarlyStopper,
    RunningStatAverager,
    _logits,
    formatted_timestamp,
    safe_backward,
    time_delta_seconds,
)
from tqdm import tqdm
from transformers import AutoTokenizer

LOG = logging.getLogger(__name__)


class MemoryMappedLogits:
    """Memory-mapped storage for logits to reduce RAM usage"""
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.file_paths = []
        
    def append(self, tensor):
        """기존 .append()와 동일한 인터페이스"""
        idx = len(self.file_paths)
        file_path = f"{self.temp_dir}/logits_{idx}.dat"
        
        # numpy로 변환 후 메모리 매핑 파일 생성
        tensor_np = tensor.cpu().numpy().astype(np.float16)
        mmap_array = np.memmap(file_path, dtype=np.float16, mode='w+', shape=tensor_np.shape)
        mmap_array[:] = tensor_np
        mmap_array.flush()
        
        self.file_paths.append((file_path, tensor_np.shape))
        return len(self.file_paths) - 1
        
    def __getitem__(self, idx):
        """기존 list[idx]와 동일한 인터페이스"""
        file_path, shape = self.file_paths[idx]
        mmap_array = np.memmap(file_path, dtype=np.float16, mode='r', shape=shape)
        return torch.from_numpy(mmap_array[:]).cuda().float()
        
    def __len__(self):
        return len(self.file_paths)
        
    def pop(self, idx=0):
        """기존 list.pop()과 동일한 인터페이스"""
        if idx >= len(self.file_paths):
            raise IndexError("pop index out of range")
        
        # 데이터 로드
        file_path, shape = self.file_paths[idx]
        mmap_array = np.memmap(file_path, dtype=np.float16, mode='r', shape=shape)
        result = torch.from_numpy(mmap_array[:]).cuda().float()
        
        # 파일 삭제 및 리스트에서 제거
        try:
            os.remove(file_path)
        except:
            pass
        self.file_paths.pop(idx)
        
        return result
        
    def __del__(self):
        """임시 디렉토리 정리"""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass


class MultimodalTrainer(BaseTrainer):
    def __init__(self, config, train_set: Dataset, val_set: Dataset):
        super().__init__(config, train_set, val_set)

        if hasattr(self.model, "edit_lrs") and not self.config.eval_only:
            self.lr_opt = self.OptimizerClass([self.model.edit_lrs], config.lr_lr)
            if self.archive is not None:
                self.lr_opt.load_state_dict(self.archive["lr_opt"])
        else:
            self.lr_opt = None

        if hasattr(self.config, "ft"):
            if getattr(self.config.ft, "use_locality", False):
                batch = next(self.edit_gen)
                self.model.loc_ids = batch["loc"]["input_ids"]
                self.model.loc_masks = batch["loc"]["attention_mask"]

    def edit_step(self, batch, training: bool):
        self.model.train(training)
        self.original_model.train(training)

        # Do the edit
        start = time.time()
        edited_model, model_info = self.model.edit(batch["edit_inner"], batch["cond"])
        edit_time = time.time() - start

        l_total, l_edit, l_loc, l_base = 0, 0, 0, 0
        info_dict = {}

        ################ portability #################
        if batch['port'] is not None:
            port_acc = 0
            assert len(batch['port']) == 1, "batch['port'] should have only one element"
            for port in batch['port']:
                with torch.no_grad():
                    port_outputs = edited_model(port)
                    port_labels = port["labels"]
                    if not isinstance(port_outputs, torch.Tensor):
                        port_logits = port_outputs.logits
                    else:
                        port_logits = port_outputs
                    if port_logits.shape[1] > port_labels.shape[1]:
                        port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
                    else:
                        port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
                    port_acc += port_dict["acc"].item()
                    info_dict['grad/port_pred_ids'] = port_dict['pred_ids']
                    info_dict['grad/port_targ_ids'] = port_dict['targ_ids']
            port_acc /= len(batch['port'])
            info_dict['port/acc'] = port_acc
        ################ portability #################
        
        info_dict = {**info_dict, **model_info}

        return l_total, l_edit, l_loc, l_base, info_dict

    def train_step(self, batch):
        l_total, l_edit, l_loc, l_base, info_dict = self.edit_step(
            batch, training=True
        )

        if self.global_iter > 0 and self.global_iter % self.config.accumulate_bs == 0:
            grad = torch.nn.utils.clip_grad_norm_(
                self.model.outer_parameters(),
                self.config.grad_clip,
                error_if_nonfinite=True,
            )
            info_dict['grad'] = grad.item()

            self.opt.step()
            self.opt.zero_grad()

            if self.lr_opt is not None:
                self.lr_opt.step()
                self.lr_opt.zero_grad()

                for lr_idx, lr in enumerate(self.model.edit_lrs):
                    info_dict[f'lr/lr{lr_idx}'] = lr.item()

        return info_dict

    def _inline_validation_log(self, step, stats, start_time, steps):
        elapsed = (time.time() - start_time) / (step + 1)
        prog = f"{step+1}/{steps}".ljust(20)
        #################################################################
        # inner_acc = f"{stats['inner/acc_val']:<12.5f}"
        # outer_acc = f"{stats['edit/acc_val']:<12.5f}"
        # image_acc = f"{stats['image_rephrase/acc_val']:<12.5f}"
        # loc_acc = f"{stats['loc/acc_val']:<12.5f}"
        # loc_image_acc = f"{stats['image_loc/acc_val']:<12.5f}"

        # LOG.info(
        #   f"Step {prog} outer_acc: {outer_acc} image_acc: {image_acc} inner_acc: {inner_acc} it_time: {elapsed:.4f} loc_acc: {loc_acc}, image_loc: {loc_image_acc}"
        # )
        #################################################################
        if 'port/acc_val' in stats:
            LOG.info(f"step {prog} port_acc: {stats['port/acc_val']:<12.5f} it_time: {elapsed:.4f}")
       
        if 'knowledge/acc_val' in stats:
            LOG.info(f"step {prog} knowledge_acc: {stats['knowledge/acc_val']:<12.5f} it_time: {elapsed:.4f}")

    def validate(self, steps=None, log: bool = False, result_name: str = None):
        if steps is None or steps > len(self.val_set):
            steps = len(self.val_set)

        if log:
            LOG.info(f"Beginning evaluation for {steps} steps...")
        averager = RunningStatAverager("val")

        start_time = time.time()
        if result_name is not None:
            port_result = []
        for val_step, batch in tqdm(enumerate(self.val_loader), total=steps, desc="Validation", ncols=100):
            if val_step >= steps:
                break
            
            if (log and (val_step) % self.config.log_interval == 0):
                self._inline_validation_log(
                    val_step, averager.average(), start_time, steps
                )
           
            if batch['port'] is None:
                continue
            _, _, _, _, info_dict = self.edit_step(batch, training=False)
            averager.add(info_dict) 

            # append write to txt file info_dict['port/acc']
            if result_name is not None:
                edit_inputs = batch['edit_inner']['text_input']
                port_inputs = batch['port'][0]['text_input']
                port_acc = info_dict['port/acc']
                port_pred_ids = info_dict['grad/port_pred_ids'].cpu().numpy()
                port_targ_ids = info_dict['grad/port_targ_ids'].cpu().numpy()
                # with open(f'results/results_multihop/{result_name}_port_hop{self.val_set.hop}.txt', 'a') as f:
                #     f.write(f'{edit_inputs}\n{port_inputs}\n{port_acc}\npred: {port_pred_ids}\ntarget: {port_targ_ids}\n\n')
                port_result.append({
                    'edit_input': edit_inputs,
                    'port_input': port_inputs,
                    'port_acc': port_acc,
                    'port_pred_ids': port_pred_ids.tolist(),
                    'port_targ_ids': port_targ_ids.tolist()
                })
        
        if result_name is not None:
            with open(f'results/results_multihop/{result_name}_port_hop{self.val_set.hop}.json', 'w') as f:
                json.dump(port_result, f, indent=2)

        if log:
            self._inline_validation_log(val_step, averager.average(), start_time, steps)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        return stats
    
    def knowledge_qa(self, batch, training: bool):
        self.model.train(training)
        self.original_model.train(training)

        l_total, l_edit, l_loc, l_base = 0, 0, 0, 0
        info_dict = {}

        assert batch['port'] is not None, "portability edit must be provided"
        assert len(batch['port']) == 1, "batch['port'] should have only one element"

        knowledge = batch['port'][0]
        with torch.no_grad():
            knowledge_outputs = self.model(knowledge)
            knowledge_labels = knowledge["labels"]
            if not isinstance(knowledge_outputs, torch.Tensor):
                knowledge_logits = knowledge_outputs.logits
            else:
                knowledge_logits = knowledge_outputs
            if knowledge_logits.shape[1] > knowledge_labels.shape[1]:
                knowledge_dict = self.model.edit_loss_fn(self.config, knowledge_logits, knowledge_labels)
            else:
                knowledge_dict = self.model.edit_loss_fn(self.config, knowledge_logits, knowledge_labels[:, -knowledge_logits.shape[1]-1:])
            knowledge_acc = knowledge_dict["acc"].item()
        info_dict['knowledge/acc'] = knowledge_acc
        
        info_dict = {**info_dict, **{}}

        return l_total, l_edit, l_loc, l_base, info_dict
    
    def test_knowledge(self, steps=None, log: bool = False):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.eval()

        if steps is None or steps > len(self.val_set):
            steps = len(self.val_set)

        if log:
            LOG.info(f"Beginning evaluation for {steps} steps...")
        averager = RunningStatAverager("val")

        start_time = time.time()
        for val_step, batch in tqdm(enumerate(self.val_loader), total=steps, desc="Validation", ncols=100):
            if val_step >= steps:
                break
            
            if (log and (val_step) % self.config.log_interval == 0):
                self._inline_validation_log(
                    val_step, averager.average(), start_time, steps
                )

            _, _, _, _, info_dict = self.knowledge_qa(batch, training=False)
            averager.add(info_dict) 

        if log:
            self._inline_validation_log(val_step, averager.average(), start_time, steps)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps


        results_path = f"results/results_base_port/{cur_time}_{self.config.model_name}_port{self.val_set.hop}_questiontest.json"

        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats

    def _inline_seq_log(self, step, stats, start_time, steps):
        elapsed = (time.time() - start_time) / (step + 1)
        prog = f"{step+1}/{steps}".ljust(20)
        inner_acc = f"{stats['inner/acc_val']:<12.5f}"
        outer_acc = f"{stats['edit/acc_val']:<12.5f}"
        image_acc = f"{stats['image_rephrase/acc_val']:<12.5f}"
        loc_acc = f"{stats['loc/acc_val']:<12.5f}"
        loc_image_acc = f"{stats['image_loc/acc_val']:<12.5f}"
        port_acc = f"{stats['port/acc_val']:<12.5f}"
        LOG.info(
          f"Step {prog} outer_acc: {outer_acc} image_acc: {image_acc} inner_acc: {inner_acc} it_time: {elapsed:.4f} loc_acc: {loc_acc}, image_loc: {loc_image_acc}, port_acc: {port_acc}"
        )
    # for textual edit
    def _inline_seq_log_textualEdit(self, step, stats, start_time, steps):
        elapsed = (time.time() - start_time) / (step + 1)
        prog = f"{step+1}/{steps}".ljust(20)
        inner_acc = f"{stats['inner/acc_val']:<12.5f}"
        outer_acc = f"{stats['edit/acc_val']:<12.5f}"
        loc_acc = f"{stats['loc/acc_val']:<12.5f}"
        LOG.info(
          f"Step {prog} outer_acc: {outer_acc}  inner_acc: {inner_acc} it_time: {elapsed:.4f} loc_acc: {loc_acc}"
        )
    
    # for compositional Edit
    def _inline_seq_log_CompositionalEdit(self, step, stats, start_time, steps):
        elapsed = (time.time() - start_time) / (step + 1)
        prog = f"{step+1}/{steps}".ljust(20)
        # visual
        v_inner_acc = f"{stats['vis/inner/acc_val']:<12.5f}"
        v_outer_acc = f"{stats['vis/edit/acc_val']:<12.5f}"
        v_image_acc = f"{stats['vis/image_rephrase/acc_val']:<12.5f}"
        v_loc_acc = f"{stats['vis/loc/acc_val']:<12.5f}"
        v_loc_image_acc = f"{stats['vis/image_loc/acc_val']:<12.5f}"

        # textual
        t_inner_acc = f"{stats['text/inner/acc_val']:<12.5f}"
        t_outer_acc = f"{stats['text/edit/acc_val']:<12.5f}"
        t_loc_acc = f"{stats['text/loc/acc_val']:<12.5f}"
        
        # compositional
        port_acc = f"{stats['port/acc_val']:<12.5f}"
        if self.config.for_eval:
            port_ratio = f"{stats['port/ratio_val']:<12.5f}"
            LOG.info(
                f"Step {prog} | "
                f"[Visual Edit] - inner_acc: {v_inner_acc} outer_acc: {v_outer_acc} img_acc: {v_image_acc}| "
                f"loc_acc: {v_loc_acc} img_loc_acc: {v_loc_image_acc}                                     | "
                f"[Textual Edit] - inner_acc: {t_inner_acc} outer_acc: {t_outer_acc} loc_acc: {t_loc_acc} | "
                f"[Compositional Edit] - Port_acc: {port_acc} Port_ratio: {port_ratio}                    | "                       
                f"it_time: {elapsed:.4f}s"
            )
        else:
            LOG.info(
                f"Step {prog} | "
                f"[Visual Edit] - inner_acc: {v_inner_acc} outer_acc: {v_outer_acc} img_acc: {v_image_acc}| "
                f"loc_acc: {v_loc_acc} img_loc_acc: {v_loc_image_acc}                                     | "
                f"[Textual Edit] - inner_acc: {t_inner_acc} outer_acc: {t_outer_acc} loc_acc: {t_loc_acc} | "
                f"[Compositional Edit] - Port_acc: {port_acc}                                             | " 
                f"it_time: {elapsed:.4f}s"
            )

    ## TEST(실제 inference, acc 측정)
    def test_sequencial_step(self, batch, edited_model, base_logits, base_image_logits):
        info_dict = {}

        ##############################################################################
        with torch.no_grad():
            # inner(Rel?)
            inner_edit_outputs = edited_model(batch["edit_inner"])
            inner_batch_labels = batch["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(T-Gen)
            post_edit_outputs = edited_model(batch["edit_outer"])
            post_batch_labels = batch["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # image rephrase(I-Gen)
            post_image_edit_outputs = edited_model(batch["edit_outer_image"])
            post_image_batch_labels = batch["edit_outer_image"]["labels"]
            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
            else:
                post_image_edit_logits = post_image_edit_outputs

            if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)
            else:
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])
            del post_image_edit_outputs, post_image_edit_logits
            torch.cuda.empty_cache()

            # text loc(T-Loc)
            post_base_outputs = edited_model(batch["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

            # image loc(I-Loc)
            post_image_base_outputs = edited_model(batch["loc_image"])
            if not isinstance(post_image_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
            else:
                post_image_base_logits = post_image_base_outputs
            post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
            base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10, dim=-1).indices
            del post_image_base_outputs, post_image_base_logits
            torch.cuda.empty_cache()

        info_dict['inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['edit/acc'] = post_edit_dict["acc"].item()
        info_dict['image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ################ portability #################
        assert len(batch['port']) == 1, "batch['port'] exist and have only one element"
        port = batch['port'][0]
        with torch.no_grad():
            port_outputs = edited_model(port)
            port_labels = port["labels"]
            if not isinstance(port_outputs, torch.Tensor):
                port_logits = port_outputs.logits
            else:
                port_logits = port_outputs
            if port_logits.shape[1] > port_labels.shape[1]:
                port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
            else:
                port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
            port_acc = port_dict["acc"].item()
        info_dict['port/acc'] = port_acc
        ################ portability #################

        return info_dict
    def test_sequencial(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") # 궁금한게, 200개에 대한 batch
        averager = RunningStatAverager("val")

        start_time = time.time()
        val_data_store = []
        base_logits_store = []
        base_image_logits_store = []
        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        # 여기선 뭘 준비하는거지? test num 200 만큼 데이터, inference 저장
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num: # 우선, Locality 평가를 위해 test_num(=200개)만큼, batch output을 뽑아냄(T-Loc, I-Loc) -> 각 출력을 저장
                val_data_store.append(batch) # batch 데이터 저장
                with torch.no_grad():
                    base_outputs = self.model(batch["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["loc_image"]) # I-Loc inference 저장
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store.append(base_image_logits.clone().detach())
                pbar.update(1)
            else:
                break
        pbar.close()

        # 여기선 뭘 하는거지? Model edit -> Test
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            # 1. Model Edit (Update for a batch)
            edited_model, _ = edited_model.edit(batch["edit_inner"], batch["cond"], detach_history=True)

            # 2. Test with GAP(?)
            if val_step >= gap_num: # if gap 10,  1 >= 10 -> 2 >= 10, 3 >= 10 ...(edit) ->  10 >= 10(gap), then test
                # 기존 저장했던 batch, t-loc & i-loc-logit 불러옴. For Test
                stored_batch = val_data_store.pop(0)
                stored_base_logits = base_logits_store.pop(0)
                stored_base_image_logits = base_image_logits_store.pop(0)

                # Test Sequential Edit 
                info_dict = self.test_sequencial_step(stored_batch, edited_model, stored_base_logits, stored_base_image_logits)
                averager.add(info_dict)

            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log(
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        if log:
            self._inline_seq_log(val_step, averager.average(), start_time, steps)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        results_path = f"results/results_sequencial/visual_edit/{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats

    def test_sequencial_compositional_ft(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") # 궁금한게, 200개에 대한 batch
        averager = RunningStatAverager("val")

        start_time = time.time()
        ## 저장할 내용
        val_data_store = []

        # visul-data
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        # textual-data
        base_logits_store_tex = []

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 1. Inference Output for test locality(visual & textual 둘다 저장 & 출력)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                # 1.1) visual edit part
                val_data_store.append(batch) # batch 데이터 저장
                with torch.no_grad():
                    base_outputs = self.model(batch["visual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["visual_edit"]["loc_image"]) # I-Loc inference 저장
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach())

                # 1.2) textual edit part
                with torch.no_grad():
                    base_outputs = self.model(batch["textual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_tex.append(base_logits.clone().detach())

                pbar.update(1)
            else:
                break
        pbar.close()

        ## 2. Model edit & Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            # 2.1) Model Edit (Update for a batch)
            # 2.1.1) Visual Edit(first)
            edited_model, _ = edited_model.edit(batch["visual_edit"]["edit_inner"], batch["cond"], detach_history=True)

            # 2.1.2) Textual Edit(second) ★☆★☆★☆ --> __getitem__ 시 확인. collate_fn
            edited_model, _ = edited_model.edit(batch["textual_edit"]["edit_inner"], batch["cond"], detach_history=True)
            ## **순서상 Textual Edit이 나중에 되고, 바로 평가가 되니 textual edit이 조금이나마 더 잘나오지 않을까 생각 ##

            # 2.2) Test with GAP
            if val_step >= gap_num: 
                # 기존 저장했던 batch, t-loc & i-loc-logit 불러옴. For Test
                stored_batch = val_data_store.pop(0) # vis + text
                stored_base_logits_vis = base_logits_store_vis.pop(0)
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0)
                stored_base_logits_tex = base_logits_store_tex.pop(0)

                # Test Sequential Edit(only inference & test) - vis / text 모두 다 평가해야 함.
                info_dict = self.test_sequencial_compositional_step(
                    stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_tex
                    )
                averager.add(info_dict)

            # logging?
            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log_CompositionalEdit( ## ★☆★ 수정 필요 ★☆★ ##
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Logging Results ## 
        if log:
            self._inline_seq_log_CompositionalEdit(val_step, averager.average(), start_time, steps) ## ★☆★ 수정 필요 ★☆★ ##
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        if 'llava' in self.config.model_name.lower():
            if os.path.basename(self.config.name) == "llava-v1.5-13b":
                print("-> llava 13B 저장중...")
                result_dir = f"results/results_sequencial/llava1.5v_13b/composition/ft"
            elif os.path.basename(self.config.name) == "llava-v1.5-7b":
                result_dir = f"results/results_sequencial/composition/ft"
                print("-> llava 7B 저장중...")
        elif 'blip2' in self.config.model_name.lower() :
            result_dir = f"results/results_sequencial/blip2/composition/ft"
        elif 'minigpt4' in self.config.model_name.lower() :
            result_dir = f"results/results_sequencial/minigpt4/composition/ft"
        
        results_path = os.path.join(result_dir, f"{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json")
        

        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats
    

        ## TEST - compositonal
    
        ## TEST - compositonal step(실제 inference, acc 측정)
    def test_sequencial_compositional_ft_step(self, batch, edited_model, base_logits_vis, base_image_logits_vis, base_logits_tex):
        info_dict = {}

        ##############################################################################
        # ----------------------------Test: Visual Edit------------------------------#
        with torch.no_grad():
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["visual_edit"]["edit_inner"])
            inner_batch_labels = batch["visual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels) # edit_loss_fn이 어딨지?
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(T-Gen)
            post_edit_outputs = edited_model(batch["visual_edit"]["edit_outer"])
            post_batch_labels = batch["visual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels) # edit_loss_fn -> vis, text 한번에 적용해도 되는가
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # image rephrase(I-Gen)
            post_image_edit_outputs = edited_model(batch["visual_edit"]["edit_outer_image"])
            post_image_batch_labels = batch["visual_edit"]["edit_outer_image"]["labels"]
            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
            else:
                post_image_edit_logits = post_image_edit_outputs

            if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)
            else:
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])
            del post_image_edit_outputs, post_image_edit_logits
            torch.cuda.empty_cache()

            # text loc(T-Loc)
            post_base_outputs = edited_model(batch["visual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_vis, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

            # image loc(I-Loc)
            post_image_base_outputs = edited_model(batch["visual_edit"]["loc_image"])
            if not isinstance(post_image_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
            else:
                post_image_base_logits = post_image_base_outputs
            post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
            base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits_vis, dim=-1), k=10, dim=-1).indices
            del post_image_base_outputs, post_image_base_logits
            torch.cuda.empty_cache()

        info_dict['vis/inner/acc'] = inner_edit_dict["acc"].item() # copy안해도 되는가? -> item은 int/float이 직접 반환됨. 따라서 ㅇㅇ
        info_dict['vis/edit/acc'] = post_edit_dict["acc"].item()
        info_dict['vis/image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["vis/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["vis/image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ##############################################################################
        # ----------------------------Test: Textual Edit------------------------------#
        with torch.no_grad():
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["textual_edit"]["edit_inner"])
            inner_batch_labels = batch["textual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(Generality)
            post_edit_outputs = edited_model(batch["textual_edit"]["edit_outer"])
            post_batch_labels = batch["textual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # text loc(Locality)
            post_base_outputs = edited_model(batch["textual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_tex, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

        info_dict['text/inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['text/edit/acc'] = post_edit_dict["acc"].item()
        info_dict["text/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ################ portability #################
        if batch['port'] is not None:
            assert len(batch['port']) == 1, "batch['port'] exist and have only one element"
            port = batch['port'][0]
            with torch.no_grad():
                port_outputs = edited_model(port)
                port_labels = port["labels"]
                if not isinstance(port_outputs, torch.Tensor):
                    port_logits = port_outputs.logits
                else:
                    port_logits = port_outputs
                if port_logits.shape[1] > port_labels.shape[1]:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
                else:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
                port_acc = port_dict["acc"].item()
            info_dict['port/acc'] = port_acc
            ################ portability #################

        return info_dict

    def test_sequencial_compositional_ft_vis(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") # 궁금한게, 200개에 대한 batch
        averager = RunningStatAverager("val")

        start_time = time.time()
        ## 저장할 내용
        val_data_store = []

        # visul-data
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        # textual-data
        base_logits_store_tex = []

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 1. Inference Output for test locality(visual & textual 둘다 저장 & 출력)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                # 1.1) visual edit part
                val_data_store.append(batch) # batch 데이터 저장
                with torch.no_grad():
                    base_outputs = self.model(batch["visual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["visual_edit"]["loc_image"]) # I-Loc inference 저장
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach())

                # 1.2) textual edit part
                with torch.no_grad():
                    base_outputs = self.model(batch["textual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_tex.append(base_logits.clone().detach())

                pbar.update(1)
            else:
                break
        pbar.close()

        ## 2. Model edit & Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            # 2.1) Model Edit (Update for a batch)
            # 2.1.1) Visual Edit(first)
            edited_model, _ = edited_model.edit(batch["visual_edit"]["edit_inner"], batch["cond"], detach_history=True)

            # 2.1.2) Textual Edit(second) ★☆★☆★☆ --> __getitem__ 시 확인. collate_fn
            edited_model, _ = edited_model.edit(batch["textual_edit"]["edit_inner"], batch["cond"], detach_history=True)
            ## **순서상 Textual Edit이 나중에 되고, 바로 평가가 되니 textual edit이 조금이나마 더 잘나오지 않을까 생각 ##

            # 2.2) Test with GAP
            if val_step >= gap_num: 
                # 기존 저장했던 batch, t-loc & i-loc-logit 불러옴. For Test
                stored_batch = val_data_store.pop(0) # vis + text
                stored_base_logits_vis = base_logits_store_vis.pop(0)
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0)
                stored_base_logits_tex = base_logits_store_tex.pop(0)

                # Test Sequential Edit(only inference & test) - vis / text 모두 다 평가해야 함.
                info_dict = self.test_sequencial_compositional_step(
                    stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_tex
                    )
                averager.add(info_dict)

            # logging?
            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log_CompositionalEdit( ## ★☆★ 수정 필요 ★☆★ ##
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Logging Results ## 
        if log:
            self._inline_seq_log_CompositionalEdit(val_step, averager.average(), start_time, steps) ## ★☆★ 수정 필요 ★☆★ ##
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        results_path = f"results/results_sequencial/composition/ft_vis/{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json"
        
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats
    

        ## TEST - compositonal
    
        ## TEST - compositonal step(실제 inference, acc 측정)
    def test_sequencial_compositional_ft_vis_step(self, batch, edited_model, base_logits_vis, base_image_logits_vis, base_logits_tex):
        info_dict = {}

        ##############################################################################
        # ----------------------------Test: Visual Edit------------------------------#
        with torch.no_grad():
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["visual_edit"]["edit_inner"])
            inner_batch_labels = batch["visual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels) # edit_loss_fn이 어딨지?
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(T-Gen)
            post_edit_outputs = edited_model(batch["visual_edit"]["edit_outer"])
            post_batch_labels = batch["visual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels) # edit_loss_fn -> vis, text 한번에 적용해도 되는가
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # image rephrase(I-Gen)
            post_image_edit_outputs = edited_model(batch["visual_edit"]["edit_outer_image"])
            post_image_batch_labels = batch["visual_edit"]["edit_outer_image"]["labels"]
            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
            else:
                post_image_edit_logits = post_image_edit_outputs

            if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)
            else:
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])
            del post_image_edit_outputs, post_image_edit_logits
            torch.cuda.empty_cache()

            # text loc(T-Loc)
            post_base_outputs = edited_model(batch["visual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_vis, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

            # image loc(I-Loc)
            post_image_base_outputs = edited_model(batch["visual_edit"]["loc_image"])
            if not isinstance(post_image_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
            else:
                post_image_base_logits = post_image_base_outputs
            post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
            base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits_vis, dim=-1), k=10, dim=-1).indices
            del post_image_base_outputs, post_image_base_logits
            torch.cuda.empty_cache()

        info_dict['vis/inner/acc'] = inner_edit_dict["acc"].item() # copy안해도 되는가? -> item은 int/float이 직접 반환됨. 따라서 ㅇㅇ
        info_dict['vis/edit/acc'] = post_edit_dict["acc"].item()
        info_dict['vis/image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["vis/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["vis/image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ##############################################################################
        # ----------------------------Test: Textual Edit------------------------------#
        with torch.no_grad():
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["textual_edit"]["edit_inner"])
            inner_batch_labels = batch["textual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(Generality)
            post_edit_outputs = edited_model(batch["textual_edit"]["edit_outer"])
            post_batch_labels = batch["textual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # text loc(Locality)
            post_base_outputs = edited_model(batch["textual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_tex, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

        info_dict['text/inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['text/edit/acc'] = post_edit_dict["acc"].item()
        info_dict["text/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ################ portability #################
        if batch['port'] is not None:
            assert len(batch['port']) == 1, "batch['port'] exist and have only one element"
            port = batch['port'][0]
            with torch.no_grad():
                port_outputs = edited_model(port)
                port_labels = port["labels"]
                if not isinstance(port_outputs, torch.Tensor):
                    port_logits = port_outputs.logits
                else:
                    port_logits = port_outputs
                if port_logits.shape[1] > port_labels.shape[1]:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
                else:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
                port_acc = port_dict["acc"].item()
            info_dict['port/acc'] = port_acc
            ################ portability #################

        return info_dict


    ## TEST - textual step(실제 inference, acc 측정)
    def test_sequencial_textual_step(self, batch, edited_model, base_logits):
        info_dict = {}

        ##############################################################################
        with torch.no_grad():
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["edit_inner"])
            inner_batch_labels = batch["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(Generality)
            post_edit_outputs = edited_model(batch["edit_outer"])
            post_batch_labels = batch["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # text loc(Locality)
            post_base_outputs = edited_model(batch["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

        info_dict['inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['edit/acc'] = post_edit_dict["acc"].item()
        info_dict["loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ################ portability #################
        if batch['port'] is not None:
            assert len(batch['port']) == 1, "batch['port'] exist and have only one element"
            port = batch['port'][0]
            with torch.no_grad():
                port_outputs = edited_model(port)
                port_labels = port["labels"]
                if not isinstance(port_outputs, torch.Tensor):
                    port_logits = port_outputs.logits
                else:
                    port_logits = port_outputs
                if port_logits.shape[1] > port_labels.shape[1]:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
                else:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
                port_acc = port_dict["acc"].item()
            info_dict['port/acc'] = port_acc
            ################ portability #################

        return info_dict
    def test_sequencial_textual(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") # 궁금한게, 200개에 대한 batch
        averager = RunningStatAverager("val")

        start_time = time.time()
        val_data_store = []
        base_logits_store = []
        #base_image_logits_store = []
        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 우선, Locality 평가를 위해 test_num(=200개)만큼, batch output을 뽑아냄(T-Loc, I-Loc) -> 각 출력을 저장 ##
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num: 
                val_data_store.append(batch) # batch 데이터 저장
                with torch.no_grad():
                    base_outputs = self.model(batch["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store.append(base_logits.clone().detach())
                    #torch.cuda.empty_cache()

                pbar.update(1)
            else:
                break
        pbar.close()

        ## Model edit -> Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            # 1. Model Edit (Update for a batch)
            edited_model, _ = edited_model.edit(batch["edit_inner"], batch["cond"], detach_history=True) 

            # 2. Test with GAP(?)
            if val_step >= gap_num: 
                # 기존 저장했던 batch, t-loc & i-loc-logit 불러옴. For Test
                stored_batch = val_data_store.pop(0)
                stored_base_logits = base_logits_store.pop(0)

                # Test Sequential Edit  # # 여기서 gpu 0번 잡힌다. 다시 4로 보내주면될듯
                info_dict = self.test_sequencial_textual_step(stored_batch, edited_model, stored_base_logits)
                averager.add(info_dict)

            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log_textualEdit(
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Logging Results ## 
        if log:
            self._inline_seq_log_textualEdit(val_step, averager.average(), start_time, steps)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        results_path = f"results/results_sequencial/textual_edit/{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}.json"

        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats
    
        ## TEST - vis + textual
    
    ###################################################################
    ####### ------------Sequential + @ ----------------------------------#####
    ## TEST - compositonal step(실제 inference, acc 측정)   
    def test_sequencial_compositional(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") # 궁금한게, 200개에 대한 batch
        averager = RunningStatAverager("val")

        start_time = time.time()
        ## 저장할 내용
        val_data_store = []

        # visul-data
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        # textual-data
        base_logits_store_tex = []

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 1. Inference Output for test locality(visual & textual 둘다 저장 & 출력)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                # 1.1) visual edit part
                val_data_store.append(batch) # batch 데이터 저장
                with torch.no_grad():
                    base_outputs = self.model(batch["visual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["visual_edit"]["loc_image"]) # I-Loc inference 저장
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach())

                # 1.2) textual edit part
                with torch.no_grad():
                    base_outputs = self.model(batch["textual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_tex.append(base_logits.clone().detach())

                pbar.update(1)
            else:
                break
        pbar.close()

        ## 2. Model edit & Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            # 2.1) Model Edit (Update for a batch)
            # 2.1.1) Visual Edit(first)
            edited_model, _ = edited_model.edit(batch["visual_edit"]["edit_inner"], batch["cond"], detach_history=True)

            # 2.1.2) Textual Edit(second) ★☆★☆★☆ --> __getitem__ 시 확인. collate_fn
            edited_model, _ = edited_model.edit(batch["textual_edit"]["edit_inner"], batch["cond"], detach_history=True)
            ## **순서상 Textual Edit이 나중에 되고, 바로 평가가 되니 textual edit이 조금이나마 더 잘나오지 않을까 생각 ##

            # 2.2) Test with GAP
            if val_step >= gap_num: 
                # 기존 저장했던 batch, t-loc & i-loc-logit 불러옴. For Test
                stored_batch = val_data_store.pop(0) # vis + text
                stored_base_logits_vis = base_logits_store_vis.pop(0)
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0)
                stored_base_logits_tex = base_logits_store_tex.pop(0)

                # Test Sequential Edit(only inference & test) - vis / text 모두 다 평가해야 함.
                info_dict = self.test_sequencial_compositional_step(
                    stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_tex
                    )
                averager.add(info_dict)

            # logging?
            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log_CompositionalEdit( ## ★☆★ 수정 필요 ★☆★ ##
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Logging Results ## 
        if log:
            self._inline_seq_log_CompositionalEdit(val_step, averager.average(), start_time, steps) ## ★☆★ 수정 필요 ★☆★ ##
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        if 'llava' in self.config.model_name.lower():
            result_dir = f"results/results_sequencial/composition/lora"
        elif 'blip2' in self.config.model_name.lower() :
            result_dir = f"results/results_sequencial/blip2/composition/lora"

        results_path = os.path.join(result_dir, f"{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json")

        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats
    

        ## TEST - compositonal
    
        ## TEST - compositonal step(실제 inference, acc 측정)
    def test_sequencial_compositional_step(self, batch, edited_model, base_logits_vis, base_image_logits_vis, base_logits_tex):
        info_dict = {}

        ##############################################################################
        # ----------------------------Test: Visual Edit------------------------------#
        with torch.no_grad():
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["visual_edit"]["edit_inner"])
            inner_batch_labels = batch["visual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels) # edit_loss_fn이 어딨지?
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(T-Gen)
            post_edit_outputs = edited_model(batch["visual_edit"]["edit_outer"])
            post_batch_labels = batch["visual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels) # edit_loss_fn -> vis, text 한번에 적용해도 되는가
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # image rephrase(I-Gen)
            post_image_edit_outputs = edited_model(batch["visual_edit"]["edit_outer_image"])
            post_image_batch_labels = batch["visual_edit"]["edit_outer_image"]["labels"]
            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
            else:
                post_image_edit_logits = post_image_edit_outputs

            if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)
            else:
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])
            del post_image_edit_outputs, post_image_edit_logits
            torch.cuda.empty_cache()

            # text loc(T-Loc)
            post_base_outputs = edited_model(batch["visual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_vis, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

            # image loc(I-Loc)
            post_image_base_outputs = edited_model(batch["visual_edit"]["loc_image"])
            if not isinstance(post_image_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
            else:
                post_image_base_logits = post_image_base_outputs
            post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
            base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits_vis, dim=-1), k=10, dim=-1).indices
            del post_image_base_outputs, post_image_base_logits
            torch.cuda.empty_cache()

        info_dict['vis/inner/acc'] = inner_edit_dict["acc"].item() # copy안해도 되는가? -> item은 int/float이 직접 반환됨. 따라서 ㅇㅇ
        info_dict['vis/edit/acc'] = post_edit_dict["acc"].item()
        info_dict['vis/image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["vis/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["vis/image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ##############################################################################
        # ----------------------------Test: Textual Edit------------------------------#
        with torch.no_grad():
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["textual_edit"]["edit_inner"])
            inner_batch_labels = batch["textual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(Generality)
            post_edit_outputs = edited_model(batch["textual_edit"]["edit_outer"])
            post_batch_labels = batch["textual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # text loc(Locality)
            post_base_outputs = edited_model(batch["textual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_tex, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

        info_dict['text/inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['text/edit/acc'] = post_edit_dict["acc"].item()
        info_dict["text/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ################ portability #################
        if batch['port'] is not None:
            assert len(batch['port']) == 1, "batch['port'] exist and have only one element"
            port = batch['port'][0]
            with torch.no_grad():
                port_outputs = edited_model(port)
                port_labels = port["labels"]
                if not isinstance(port_outputs, torch.Tensor):
                    port_logits = port_outputs.logits
                else:
                    port_logits = port_outputs
                if port_logits.shape[1] > port_labels.shape[1]:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
                else:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
                port_acc = port_dict["acc"].item()
            info_dict['port/acc'] = port_acc
            ################ portability #################

        return info_dict

    ## TEST - compositonal - two lora(comp edit 비교용) ★★★
    def test_sequencial_compositional_two(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") # 궁금한게, 200개에 대한 batch
        averager = RunningStatAverager("val")

        start_time = time.time()
        ## 저장할 내용
        val_data_store = []

        # visul-data
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        # textual-data
        base_logits_store_tex = []

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 1. Inference Output for test locality(visual & textual 데이터/출력 저장 & 출력)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                # 1.1) visual edit part
                val_data_store.append(batch) # batch 데이터 저장
                with torch.no_grad():
                    base_outputs = self.model(batch["visual_edit"]["loc"]) # T-Loc inference 저장 # self.model -> ft
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["visual_edit"]["loc_image"]) # I-Loc inference 저장
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach())

                # 1.2) textual edit part
                with torch.no_grad():
                    base_outputs = self.model(batch["textual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_tex.append(base_logits.clone().detach())

                pbar.update(1)
            else:
                break
        pbar.close()

        ## 2. Model edit & Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            # 2.1) Model Edit (Update for a batch)
            # 2.1.1) Visual Edit(first)
            self.model.model.set_adapter("visual") # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["visual_edit"]["edit_inner"], mode = "visual" , peft = True)
            # 필요시, edit 후 바로 성능 체크

            # 2.1.2) Textual Edit(second) ★☆★☆★☆ --> __getitem__ 시 확인. collate_fn
            self.model.model.set_adapter("textual")  # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["textual_edit"]["edit_inner"], mode = "textual" , peft = True)

            # 2.2) Test with GAP
            if val_step >= gap_num: 
                # 기존 저장했던 batch, t-loc & i-loc-logit 불러옴. For Test
                stored_batch = val_data_store.pop(0) # vis + text
                stored_base_logits_vis = base_logits_store_vis.pop(0)
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0)
                stored_base_logits_tex = base_logits_store_tex.pop(0)

                # Test Sequential Edit(only inference & test) - vis / text 모두 다 평가해야 함.
                info_dict = self.test_sequencial_compositional_two_step(
                    stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_tex
                    )
                averager.add(info_dict)

            # logging?
            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log_CompositionalEdit( ## ★☆★ 수정 필요 ★☆★ ##
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Logging Results ## 
        if log:
            self._inline_seq_log_CompositionalEdit(val_step, averager.average(), start_time, steps) ## ★☆★ 수정 필요 ★☆★ ##
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        #results_path = f"results/results_sequencial/composition/two_lora/{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json"
        if 'llava' in self.config.model_name.lower():
            result_dir = f"results/results_sequencial/composition/two_lora"
        elif 'blip2' in self.config.model_name.lower() :
            result_dir = f"results/results_sequencial/blip2/composition/two_lora"

        results_path = os.path.join(result_dir, f"{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json")



        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats
    

        ## TEST - compositonal - two
    def test_sequencial_compositional_two_step(self, batch, edited_model, base_logits_vis, base_image_logits_vis, base_logits_tex):
        info_dict = {}

        ##############################################################################
        # ----------------------------Test: Visual Edit------------------------------#
        with torch.no_grad():
            # set lora: visual inference
            edited_model.model.set_adapter("visual")
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["visual_edit"]["edit_inner"])
            inner_batch_labels = batch["visual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels) # edit_loss_fn이 어딨지?
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(T-Gen)
            post_edit_outputs = edited_model(batch["visual_edit"]["edit_outer"])
            post_batch_labels = batch["visual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels) # edit_loss_fn -> vis, text 한번에 적용해도 되는가
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # image rephrase(I-Gen)
            post_image_edit_outputs = edited_model(batch["visual_edit"]["edit_outer_image"])
            post_image_batch_labels = batch["visual_edit"]["edit_outer_image"]["labels"]
            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
            else:
                post_image_edit_logits = post_image_edit_outputs

            if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)
            else:
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])
            del post_image_edit_outputs, post_image_edit_logits
            torch.cuda.empty_cache()

            # text loc(T-Loc)
            post_base_outputs = edited_model(batch["visual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_vis, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

            # image loc(I-Loc)
            post_image_base_outputs = edited_model(batch["visual_edit"]["loc_image"])
            if not isinstance(post_image_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
            else:
                post_image_base_logits = post_image_base_outputs
            post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
            base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits_vis, dim=-1), k=10, dim=-1).indices
            del post_image_base_outputs, post_image_base_logits
            torch.cuda.empty_cache()

        info_dict['vis/inner/acc'] = inner_edit_dict["acc"].item() # copy안해도 되는가? -> item은 int/float이 직접 반환됨. 따라서 ㅇㅇ
        info_dict['vis/edit/acc'] = post_edit_dict["acc"].item()
        info_dict['vis/image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["vis/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["vis/image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ##############################################################################
        # ----------------------------Test: Textual Edit------------------------------#
        with torch.no_grad():
            # set lora: visual inference
            edited_model.model.set_adapter("textual")
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["textual_edit"]["edit_inner"])
            inner_batch_labels = batch["textual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(Generality)
            post_edit_outputs = edited_model(batch["textual_edit"]["edit_outer"])
            post_batch_labels = batch["textual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # text loc(Locality)
            post_base_outputs = edited_model(batch["textual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_tex, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

        info_dict['text/inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['text/edit/acc'] = post_edit_dict["acc"].item()
        info_dict["text/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ################ portability #################
        #### Deprecated ####
        # if "merge" in edited_model.model.peft_config:
        #     edited_model.model.delete_adapter("merge")

        # weights = [1.0, 1.0]
        # adapter_name = "merge"
        # edited_model.model.add_weighted_adapter(["visual", "textual"], weights, adapter_name, combination_type="cat")
        # edited_model.model.set_adapter("merge")
        
        # set lora: visual&textual inference
        edited_model.model.set_adapter(["textual","visual"])

        if batch['port'] is not None:
            assert len(batch['port']) == 1, "batch['port'] exist and have only one element"
            port = batch['port'][0]
            with torch.no_grad():
                port_outputs = edited_model(port)
                port_labels = port["labels"]
                if not isinstance(port_outputs, torch.Tensor):
                    port_logits = port_outputs.logits
                else:
                    port_logits = port_outputs
                if port_logits.shape[1] > port_labels.shape[1]:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
                else:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
                port_acc = port_dict["acc"].item()
                del port_outputs, port_logits
                torch.cuda.empty_cache()

            info_dict['port/acc'] = port_acc
            ################ portability #################

        return info_dict

    # BLIP2
    def test_sequencial_compositional_two_parallel(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") # 궁금한게, 200개에 대한 batch
        averager = RunningStatAverager("val")

        start_time = time.time()
        ## 저장할 내용
        val_data_store = []

        # visul-data
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        # textual-data
        base_logits_store_tex = []

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 1. Inference Output for test locality(visual & textual 데이터/출력 저장 & 출력)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                # 1.1) visual edit part
                val_data_store.append(batch) # batch 데이터 저장
                with torch.no_grad():
                    base_outputs = self.model(batch["visual_edit"]["loc"]) # T-Loc inference 저장 # self.model -> ft
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["visual_edit"]["loc_image"]) # I-Loc inference 저장
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach())

                # 1.2) textual edit part
                with torch.no_grad():
                    base_outputs = self.model(batch["textual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_tex.append(base_logits.clone().detach())

                pbar.update(1)
            else:
                break
        pbar.close()

        ## 2. Model edit & Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            # 2.1) Model Edit (Update for a batch)
            # 2.1.1) Visual Edit(first)
            self.model.model.set_adapter("visual") # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["visual_edit"]["edit_inner"], mode = "visual" , peft = True)
            # 필요시, edit 후 바로 성능 체크

            # 2.1.2) Textual Edit(second) ★☆★☆★☆ --> __getitem__ 시 확인. collate_fn
            self.model.model.set_adapter("textual")  # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["textual_edit"]["edit_inner"], mode = "textual" , peft = True)

            # 2.2) Test with GAP
            if val_step >= gap_num: 
                # 기존 저장했던 batch, t-loc & i-loc-logit 불러옴. For Test
                stored_batch = val_data_store.pop(0) # vis + text
                stored_base_logits_vis = base_logits_store_vis.pop(0)
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0)
                stored_base_logits_tex = base_logits_store_tex.pop(0)

                # Test Sequential Edit(only inference & test) - vis / text 모두 다 평가해야 함.
                info_dict = self.test_sequencial_compositional_two_parallel_step(
                    stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_tex
                    )
                averager.add(info_dict)

            # logging?
            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log_CompositionalEdit( ## ★☆★ 수정 필요 ★☆★ ##
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Logging Results ## 
        if log:
            self._inline_seq_log_CompositionalEdit(val_step, averager.average(), start_time, steps) ## ★☆★ 수정 필요 ★☆★ ##
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        #results_path = f"results/results_sequencial/composition/two_lora/{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json"
        if 'llava' in self.config.model_name.lower():
            result_dir = f"results/results_sequencial/composition/two_lora"
        elif 'blip2' in self.config.model_name.lower() :
            result_dir = f"results/results_sequencial/blip2/composition/two_lora"

        results_path = os.path.join(result_dir, f"{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json")



        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        if gap_num == 0:
            try: # lora weight 저장
                self.model.model.save_pretrained(os.path.join(result_dir, weight))
                print("LoRA 모델 저장 완료")
            except:
                print("LoRA 모델 저장 실패")

        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats

        ## TEST - compositonal - two
    def test_sequencial_compositional_two_parallel_step(self, batch, edited_model, base_logits_vis, base_image_logits_vis, base_logits_tex):
        info_dict = {}

        ##############################################################################
        # ----------------------------Test: Visual Edit------------------------------#
        with torch.no_grad():
            # set lora: visual inference
            edited_model.model.set_adapter("visual")
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["visual_edit"]["edit_inner"])
            inner_batch_labels = batch["visual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels) # edit_loss_fn이 어딨지?
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(T-Gen)
            post_edit_outputs = edited_model(batch["visual_edit"]["edit_outer"])
            post_batch_labels = batch["visual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels) # edit_loss_fn -> vis, text 한번에 적용해도 되는가
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # image rephrase(I-Gen)
            post_image_edit_outputs = edited_model(batch["visual_edit"]["edit_outer_image"])
            post_image_batch_labels = batch["visual_edit"]["edit_outer_image"]["labels"]
            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
            else:
                post_image_edit_logits = post_image_edit_outputs

            if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)
            else:
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])
            del post_image_edit_outputs, post_image_edit_logits
            torch.cuda.empty_cache()

            # text loc(T-Loc)
            post_base_outputs = edited_model(batch["visual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_vis, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

            # image loc(I-Loc)
            post_image_base_outputs = edited_model(batch["visual_edit"]["loc_image"])
            if not isinstance(post_image_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
            else:
                post_image_base_logits = post_image_base_outputs
            post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
            base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits_vis, dim=-1), k=10, dim=-1).indices
            del post_image_base_outputs, post_image_base_logits
            torch.cuda.empty_cache()

        info_dict['vis/inner/acc'] = inner_edit_dict["acc"].item() # copy안해도 되는가? -> item은 int/float이 직접 반환됨. 따라서 ㅇㅇ
        info_dict['vis/edit/acc'] = post_edit_dict["acc"].item()
        info_dict['vis/image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["vis/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["vis/image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ##############################################################################
        # ----------------------------Test: Textual Edit------------------------------#
        with torch.no_grad():
            # set lora: visual inference
            edited_model.model.set_adapter("textual")
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["textual_edit"]["edit_inner"])
            inner_batch_labels = batch["textual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(Generality)
            post_edit_outputs = edited_model(batch["textual_edit"]["edit_outer"])
            post_batch_labels = batch["textual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # text loc(Locality)
            post_base_outputs = edited_model(batch["textual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_tex, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

        info_dict['text/inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['text/edit/acc'] = post_edit_dict["acc"].item()
        info_dict["text/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ################ portability #################
        # set lora: visual&textual inference
        edited_model.model.set_adapter(["textual","visual"])

        if batch['port'] is not None:
            assert len(batch['port']) == 1, "batch['port'] exist and have only one element"
            port = batch['port'][0]
            with torch.no_grad():
                port_outputs = edited_model(port)
                port_labels = port["labels"]
                if not isinstance(port_outputs, torch.Tensor):
                    port_logits = port_outputs.logits
                else:
                    port_logits = port_outputs
                if port_logits.shape[1] > port_labels.shape[1]:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
                else:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
                port_acc = port_dict["acc"].item()
                del port_outputs, port_logits
                torch.cuda.empty_cache()

            info_dict['port/acc'] = port_acc
            ################ portability #################

        return info_dict
    
    

    ## TEST - compositonal & connector(uni-adapter)
    def test_sequencial_compositional_connector_step(self, batch, edited_model, base_logits_vis, base_image_logits_vis, base_logits_tex):
        info_dict = {}

        ##############################################################################
        # ----------------------------Test: Visual Edit------------------------------#
        #set_edit_mode(self.model, "visual")
        with torch.no_grad():
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["visual_edit"]["edit_inner"])  # ☆☆☆ lora ☆☆☆
            inner_batch_labels = batch["visual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels) # edit_loss_fn이 어딨지?
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(T-Gen)
            post_edit_outputs = edited_model(batch["visual_edit"]["edit_outer"])  # ☆☆☆ lora ☆☆☆
            post_batch_labels = batch["visual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels) # edit_loss_fn -> vis, text 한번에 적용해도 되는가
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # image rephrase(I-Gen)
            post_image_edit_outputs = edited_model(batch["visual_edit"]["edit_outer_image"])  # ☆☆☆ lora ☆☆☆
            post_image_batch_labels = batch["visual_edit"]["edit_outer_image"]["labels"]
            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
            else:
                post_image_edit_logits = post_image_edit_outputs

            if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)
            else:
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])
            del post_image_edit_outputs, post_image_edit_logits
            torch.cuda.empty_cache()

            # text loc(T-Loc)
            post_base_outputs = edited_model(batch["visual_edit"]["loc"])  # ☆☆☆ lora ☆☆☆
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_vis, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

            # image loc(I-Loc)
            post_image_base_outputs = edited_model(batch["visual_edit"]["loc_image"]) # ☆☆☆ lora ☆☆☆
            if not isinstance(post_image_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
            else:
                post_image_base_logits = post_image_base_outputs
            post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
            base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits_vis, dim=-1), k=10, dim=-1).indices
            del post_image_base_outputs, post_image_base_logits
            torch.cuda.empty_cache()

        info_dict['vis/inner/acc'] = inner_edit_dict["acc"].item() # copy안해도 되는가? -> item은 int/float이 직접 반환됨. 따라서 ㅇㅇ
        info_dict['vis/edit/acc'] = post_edit_dict["acc"].item()
        info_dict['vis/image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["vis/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["vis/image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ##############################################################################
        # ----------------------------Test: Textual Edit------------------------------#
        #set_edit_mode(self.model, "textual")
        with torch.no_grad():
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["textual_edit"]["edit_inner"])  # ☆☆☆ lora ☆☆☆
            inner_batch_labels = batch["textual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(Generality)
            post_edit_outputs = edited_model(batch["textual_edit"]["edit_outer"]) # ☆☆☆ lora ☆☆☆
            post_batch_labels = batch["textual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # text loc(Locality)
            post_base_outputs = edited_model(batch["textual_edit"]["loc"]) # ☆☆☆ lora ☆☆☆
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_tex, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

        info_dict['text/inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['text/edit/acc'] = post_edit_dict["acc"].item()
        info_dict["text/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ################ portability #################
        if batch['port'] is not None:
            assert len(batch['port']) == 1, "batch['port'] exist and have only one element"
            port = batch['port'][0]
            with torch.no_grad():
                if self.config.lora_connector_type:
                    for module in self.model.modules():
                        if hasattr(module, "use_connector"):
                            module.use_connector() # (foward시 변경)lora -> lora + mlp 

                port_outputs = edited_model(port) # ☆☆☆ lora + mlp ☆☆☆
                port_labels = port["labels"]

                if self.config.lora_connector_type:
                    for module in self.model.modules():
                        if hasattr(module, "set_default"):
                            module.set_default() # (foward시 변경)lora + mlp -> lora

                if not isinstance(port_outputs, torch.Tensor):
                    port_logits = port_outputs.logits
                else:
                    port_logits = port_outputs
                if port_logits.shape[1] > port_labels.shape[1]:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
                else:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
                port_acc = port_dict["acc"].item()
            info_dict['port/acc'] = port_acc

            
            ################ portability #################

        return info_dict
    def test_sequencial_compositional_connector(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") 
        averager = RunningStatAverager("val")

        start_time = time.time()
        ## 저장할 내용
        val_data_store = []

        # visul-data
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        # textual-data
        base_logits_store_tex = []

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 1. Inference Output for test locality(visual & textual 둘다 저장 & 출력, only one adapter)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                # 1.1) visual edit part
                val_data_store.append(batch) # batch 데이터 저장
                with torch.no_grad():
                    base_outputs = self.model(batch["visual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["visual_edit"]["loc_image"]) # I-Loc inference 저장
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach())

                # 1.2) textual edit part
                with torch.no_grad():
                    base_outputs = self.model(batch["textual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_tex.append(base_logits.clone().detach())

                pbar.update(1)
            else:
                break
        pbar.close()

        ## 2. Model edit & Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            ## 2.1) Model Edit (Update for a batch)
            # 2.1.1) Visual Edit(first) ★이땐 mlp 학습 x
            edited_model, _ = edited_model.edit(batch["visual_edit"]["edit_inner"], batch["cond"], detach_history=True)
            # 필요시, edit 후 바로 성능 체크

            # 2.1.2) Textual Edit(second) ★이땐 mlp 학습 x | __getitem__ 시 확인. collate_fn
            edited_model, _ = edited_model.edit(batch["textual_edit"]["edit_inner"], batch["cond"], detach_history=True)
            ## **순서상 Textual Edit이 나중에 되고, 바로 평가가 되니 textual edit이 조금이나마 더 잘나오지 않을까 생각 ##

            # 2.1.3) Compositional Edit(second) ★ mlp 학습 o
            edited_model, _ = edited_model.edit(batch["port"][0], batch["cond"], detach_history=True, connector_mode=True) # cond? 이거 안되나


            # 2.2) Test with GAP
            if val_step >= gap_num: 
                # 기존 저장했던 batch, t-loc & i-loc-logit 불러옴. For Test
                stored_batch = val_data_store.pop(0) # vis + text
                stored_base_logits_vis = base_logits_store_vis.pop(0)
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0)
                stored_base_logits_tex = base_logits_store_tex.pop(0)

                # Test Sequential Edit(only inference & test) - vis / text 모두 다 평가해야 함.
                info_dict = self.test_sequencial_compositional_connector_step( # ☆☆☆ lora  / lora + mlp -> 출력 나누기 ☆☆☆
                    stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_tex
                    )
                averager.add(info_dict)

            # logging?
            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log_CompositionalEdit( 
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Save Models: LoRA, MLP ## 
        adapter_state = {
            k: v
            for k, v in edited_model.state_dict().items()
            if ("down_proj.mlp" in k or "up_proj.mlp" in k or "lora")  
        }
        # 저장 경로 지정
        adapter_save_path = f"results/results_sequencial/composition/lora_connector/one_{cur_time}_seqgap{gap_num}.pth"
        os.makedirs(os.path.dirname(adapter_save_path), exist_ok=True)
        torch.save(adapter_state, adapter_save_path)
        LOG.info(f"Saved updated LoRA and MLP parameters to: {adapter_save_path}")

        ## Logging Results ## 
        if log:
            self._inline_seq_log_CompositionalEdit(val_step, averager.average(), start_time, steps)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        results_path = f"results/results_sequencial/composition/lora_connector/{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats
    

        ## TEST - compositonal & connector step(실제 inference, acc 측정)
    
    ## TEST - compositonal & connector(uni-adapter)
    def test_sequencial_compositional_connector_two_step(self, batch, edited_model, base_logits_vis, base_image_logits_vis, base_logits_tex):
        info_dict = {}

        ##############################################################################
        # ----------------------------Test: Visual Edit------------------------------#
        set_edit_mode(self.model, "visual")
        with torch.no_grad():
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["visual_edit"]["edit_inner"])  # ☆☆☆ lora ☆☆☆
            inner_batch_labels = batch["visual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels) # edit_loss_fn이 어딨지?
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(T-Gen)
            post_edit_outputs = edited_model(batch["visual_edit"]["edit_outer"])  # ☆☆☆ lora ☆☆☆
            post_batch_labels = batch["visual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels) # edit_loss_fn -> vis, text 한번에 적용해도 되는가
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # image rephrase(I-Gen)
            post_image_edit_outputs = edited_model(batch["visual_edit"]["edit_outer_image"])  # ☆☆☆ lora ☆☆☆
            post_image_batch_labels = batch["visual_edit"]["edit_outer_image"]["labels"]
            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
            else:
                post_image_edit_logits = post_image_edit_outputs

            if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)
            else:
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])
            del post_image_edit_outputs, post_image_edit_logits
            torch.cuda.empty_cache()

            # text loc(T-Loc)
            post_base_outputs = edited_model(batch["visual_edit"]["loc"])  # ☆☆☆ lora ☆☆☆
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_vis, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

            # image loc(I-Loc)
            post_image_base_outputs = edited_model(batch["visual_edit"]["loc_image"]) # ☆☆☆ lora ☆☆☆
            if not isinstance(post_image_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
            else:
                post_image_base_logits = post_image_base_outputs
            post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
            base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits_vis, dim=-1), k=10, dim=-1).indices
            del post_image_base_outputs, post_image_base_logits
            torch.cuda.empty_cache()

        info_dict['vis/inner/acc'] = inner_edit_dict["acc"].item() # copy안해도 되는가? -> item은 int/float이 직접 반환됨. 따라서 ㅇㅇ
        info_dict['vis/edit/acc'] = post_edit_dict["acc"].item()
        info_dict['vis/image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["vis/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["vis/image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ##############################################################################
        # ----------------------------Test: Textual Edit------------------------------#
        set_edit_mode(self.model, "textual")
        with torch.no_grad():
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["textual_edit"]["edit_inner"])  # ☆☆☆ lora ☆☆☆
            inner_batch_labels = batch["textual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(Generality)
            post_edit_outputs = edited_model(batch["textual_edit"]["edit_outer"]) # ☆☆☆ lora ☆☆☆
            post_batch_labels = batch["textual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # text loc(Locality)
            post_base_outputs = edited_model(batch["textual_edit"]["loc"]) # ☆☆☆ lora ☆☆☆
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_tex, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

        info_dict['text/inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['text/edit/acc'] = post_edit_dict["acc"].item()
        info_dict["text/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ################ Compisitional Edit(Portability) #################
        set_edit_mode(self.model, "fusion")
        if batch['port'] is not None:
            assert len(batch['port']) == 1, "batch['port'] exist and have only one element"
            port = batch['port'][0]
            with torch.no_grad():
                if self.config.lora_connector_type:
                    for module in self.model.modules():
                        if hasattr(module, "use_connector"):
                            module.use_connector() # (foward시 변경)lora -> lora + mlp 

                port_outputs = edited_model(port) # ☆☆☆ lora + mlp ☆☆☆
                port_labels = port["labels"]

                if self.config.lora_connector_type:
                    for module in self.model.modules():
                        if hasattr(module, "set_default"):
                            module.set_default() # (foward시 변경)lora + mlp -> lora

                if not isinstance(port_outputs, torch.Tensor):
                    port_logits = port_outputs.logits
                else:
                    port_logits = port_outputs
                if port_logits.shape[1] > port_labels.shape[1]:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
                else:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
                port_acc = port_dict["acc"].item()
            info_dict['port/acc'] = port_acc

            
            ################ portability #################

        return info_dict
    def test_sequencial_compositional_connector_two(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") 
        averager = RunningStatAverager("val")

        start_time = time.time()
        ## 저장할 내용
        val_data_store = []

        # visul-data
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        # textual-data
        base_logits_store_tex = []

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 1. Inference Output for test locality(visual & textual 둘다 저장 & 출력, only one adapter)
        set_edit_mode(self.model, "default")
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                # 1.1) visual edit part
                val_data_store.append(batch) # batch 데이터 저장
                with torch.no_grad():
                    base_outputs = self.model(batch["visual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["visual_edit"]["loc_image"]) # I-Loc inference 저장
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach())

                # 1.2) textual edit part
                with torch.no_grad():
                    base_outputs = self.model(batch["textual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_tex.append(base_logits.clone().detach())

                pbar.update(1)
            else:
                break
        pbar.close()

        ## 2. Model edit & Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            ## 2.1) Model Edit (Update for a batch)
            # 2.1.1) Visual Edit(first) ★이땐 mlp 학습 x
            set_edit_mode(self.model, "visual")
            edited_model, _ = edited_model.edit(batch["visual_edit"]["edit_inner"], batch["cond"], detach_history=True, mode="visual")
            # 필요시, edit 후 바로 성능 체크

            # 2.1.2) Textual Edit(second) 
            set_edit_mode(self.model, "textual")
            edited_model, _ = edited_model.edit(batch["textual_edit"]["edit_inner"], batch["cond"], detach_history=True, mode="texutal")

            # 2.1.3) Compositional Edit(second) ★ mlp 학습 o
            set_edit_mode(self.model, "fusion")
            edited_model, _ = edited_model.edit(batch["port"][0], batch["cond"], detach_history=True, connector_mode=True) # cond? 이거 안되나


            # 2.2) Test with GAP
            if val_step >= gap_num: 
                # 기존 저장했던 batch, t-loc & i-loc-logit 불러옴. For Test
                stored_batch = val_data_store.pop(0) # vis + text
                stored_base_logits_vis = base_logits_store_vis.pop(0)
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0)
                stored_base_logits_tex = base_logits_store_tex.pop(0)

                # Test Sequential Edit(only inference & test) - vis / text 모두 다 평가해야 함.
                info_dict = self.test_sequencial_compositional_connector_two_step( # ☆☆☆ lora  / lora + mlp -> 출력 나누기 ☆☆☆
                    stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_tex
                    )
                averager.add(info_dict)

            # logging?
            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log_CompositionalEdit( 
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Save Models: LoRA, MLP ## 
        adapter_state = {
            k: v
            for k, v in edited_model.state_dict().items()
            if ("down_proj.mlp" in k or "up_proj.mlp" in k or "lora")  
        }
        # 저장 경로 지정
        adapter_save_path = f"results/results_sequencial/composition/lora_connector_two/{cur_time}_seqgap{gap_num}.pth"
        os.makedirs(os.path.dirname(adapter_save_path), exist_ok=True)
        torch.save(adapter_state, adapter_save_path)
        LOG.info(f"Saved updated LoRA and MLP parameters to: {adapter_save_path}")

        ## Logging Results ## 
        if log:
            self._inline_seq_log_CompositionalEdit(val_step, averager.average(), start_time, steps)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        results_path = f"results/results_sequencial/composition/lora_connector_two/{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats

    ###################################################################
    ####### ------------RAG + @ ----------------------------------#####
    ## TEST - BASE+ RAG -> compositonal edit
    def test_sequencial_base_with_rag(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") # 궁금한게, 200개에 대한 batch
        averager = RunningStatAverager("val")

        start_time = time.time()
        ## 저장할 내용
        val_data_store = []

        # visul-data
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        # textual-data
        base_logits_store_tex = []

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 1. Inference Output for test locality(visual & textual 둘다 저장 & 출력)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                # 1.1) visual edit part
                val_data_store.append(batch) # batch 데이터 저장
                with torch.no_grad():
                    base_outputs = self.model(batch["visual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["visual_edit"]["loc_image"]) # I-Loc inference 저장
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach())

                # 1.2) textual edit part
                with torch.no_grad():
                    base_outputs = self.model(batch["textual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_tex.append(base_logits.clone().detach())

                pbar.update(1)
            else:
                break
        pbar.close()

        ## 2. Model edit & Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            # 2.2) Test 
            if val_step >= gap_num: 
                # 기존 저장했던 batch, t-loc & i-loc-logit 불러옴. For Test
                stored_batch = val_data_store.pop(0) # vis + text
                stored_base_logits_vis = base_logits_store_vis.pop(0)
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0)
                stored_base_logits_tex = base_logits_store_tex.pop(0)

                # Test Sequential Edit(only inference & test) - vis / text 모두 다 평가해야 함.
                info_dict = self.test_sequencial_base_with_rag_step(
                    stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_tex
                    )
                averager.add(info_dict)

            # logging?
            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log_CompositionalEdit( ## ★☆★ 수정 필요 ★☆★ ##
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Logging Results ## 
        if log:
            self._inline_seq_log_CompositionalEdit(val_step, averager.average(), start_time, steps) ## ★☆★ 수정 필요 ★☆★ ##
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        #results_path = f"results/results_sequencial/composition/base_rag/{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json"
        
        if 'llava' in self.config.model_name.lower():
            result_dir = f"results/results_sequencial/composition/base_rag"
        elif 'blip2' in self.config.model_name.lower() :
            result_dir = f"results/results_sequencial/blip2/composition/base_rag"

        results_path = os.path.join(result_dir,f"{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json")
        

        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats
    

        ## TEST - compositonal
    
        ## TEST - compositonal step(실제 inference, acc 측정)
    def test_sequencial_base_with_rag_step(self, batch, edited_model, base_logits_vis, base_image_logits_vis, base_logits_tex):
        info_dict = {}

        ##############################################################################
        # ----------------------------Test: Visual Edit------------------------------#
        with torch.no_grad():
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["visual_edit"]["edit_inner"])
            inner_batch_labels = batch["visual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels) # edit_loss_fn이 어딨지?
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(T-Gen)
            post_edit_outputs = edited_model(batch["visual_edit"]["edit_outer"])
            post_batch_labels = batch["visual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels) # edit_loss_fn -> vis, text 한번에 적용해도 되는가
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # image rephrase(I-Gen)
            post_image_edit_outputs = edited_model(batch["visual_edit"]["edit_outer_image"])
            post_image_batch_labels = batch["visual_edit"]["edit_outer_image"]["labels"]
            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
            else:
                post_image_edit_logits = post_image_edit_outputs

            if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)
            else:
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])
            del post_image_edit_outputs, post_image_edit_logits
            torch.cuda.empty_cache()

            # text loc(T-Loc)
            post_base_outputs = edited_model(batch["visual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_vis, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

            # image loc(I-Loc)
            post_image_base_outputs = edited_model(batch["visual_edit"]["loc_image"])
            if not isinstance(post_image_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
            else:
                post_image_base_logits = post_image_base_outputs
            post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
            base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits_vis, dim=-1), k=10, dim=-1).indices
            del post_image_base_outputs, post_image_base_logits
            torch.cuda.empty_cache()

        info_dict['vis/inner/acc'] = inner_edit_dict["acc"].item() # copy안해도 되는가? -> item은 int/float이 직접 반환됨. 따라서 ㅇㅇ
        info_dict['vis/edit/acc'] = post_edit_dict["acc"].item()
        info_dict['vis/image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["vis/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["vis/image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ##############################################################################
        # ----------------------------Test: Textual Edit------------------------------#
        with torch.no_grad():
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["textual_edit"]["edit_inner"])
            inner_batch_labels = batch["textual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(Generality)
            post_edit_outputs = edited_model(batch["textual_edit"]["edit_outer"])
            post_batch_labels = batch["textual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # text loc(Locality)
            post_base_outputs = edited_model(batch["textual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_tex, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

        info_dict['text/inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['text/edit/acc'] = post_edit_dict["acc"].item()
        info_dict["text/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ################ portability #################
        if batch['port'] is not None:
            assert len(batch['port']) == 1, "batch['port'] exist and have only one element"
            port = batch['port'][0]
            with torch.no_grad():
                port_outputs = edited_model(port)
                port_labels = port["labels"]
                if not isinstance(port_outputs, torch.Tensor):
                    port_logits = port_outputs.logits
                else:
                    port_logits = port_outputs
                if port_logits.shape[1] > port_labels.shape[1]:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
                else:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
                port_acc = port_dict["acc"].item()
            info_dict['port/acc'] = port_acc
            ################ portability #################

        return info_dict

    ## TEST - RAG + LoRA -> compositonal edit
    def test_sequencial_rag_with_two_lora(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") # 궁금한게, 200개에 대한 batch
        averager = RunningStatAverager("val")

        start_time = time.time()
        ## 저장할 내용
        val_data_store = []

        # visul-data
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        # textual-data
        base_logits_store_tex = []

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 1. Inference Output for test locality(visual & textual 데이터/출력 저장 & 출력)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                # 1.1) visual edit part
                val_data_store.append(batch) # batch 데이터 저장
                with torch.no_grad():
                    base_outputs = self.model(batch["visual_edit"]["loc"]) # T-Loc inference 저장 # self.model -> ft
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["visual_edit"]["loc_image"]) # I-Loc inference 저장
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach())

                # 1.2) textual edit part
                with torch.no_grad():
                    base_outputs = self.model(batch["textual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_tex.append(base_logits.clone().detach())

                pbar.update(1)
            else:
                break
        pbar.close()

        ## 2. Model edit & Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            # 2.1) Model Edit (Update for a batch)
            # 2.1.1) Visual Edit(first)
            self.model.model.set_adapter("visual") # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["visual_edit"]["edit_inner"], mode = "visual" , peft = True)
            # 필요시, edit 후 바로 성능 체크

            # 2.1.2) Textual Edit(second) ★☆★☆★☆ --> __getitem__ 시 확인. collate_fn
            self.model.model.set_adapter("textual")  # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["textual_edit"]["edit_inner"], mode = "textual" , peft = True)

            # 2.2) Test with GAP
            if val_step >= gap_num: 
                # 기존 저장했던 batch, t-loc & i-loc-logit 불러옴. For Test
                stored_batch = val_data_store.pop(0) # vis + text
                stored_base_logits_vis = base_logits_store_vis.pop(0)
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0)
                stored_base_logits_tex = base_logits_store_tex.pop(0)

                # Test Sequential Edit(only inference & test) - vis / text 모두 다 평가해야 함.
                info_dict = self.test_sequencial_rag_with_two_lora_step(
                    stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_tex
                    )
                averager.add(info_dict)
                # old: test_sequencial_compositional_two_step 

            # logging?
            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log_CompositionalEdit( ## ★☆★ 수정 필요 ★☆★ ##
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Logging Results ## 
        if log:
            self._inline_seq_log_CompositionalEdit(val_step, averager.average(), start_time, steps) ## ★☆★ 수정 필요 ★☆★ ##
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

                
        if 'llava' in self.config.model_name.lower():
            result_dir = f"results/results_sequencial/composition/RAG_with_two_lora"
        elif 'blip2' in self.config.model_name.lower() :
            result_dir = f"results/results_sequencial/blip2/composition/RAG_with_two_lora"

        results_path = os.path.join(result_dir,f"{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json")

        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats  
    def test_sequencial_rag_with_two_lora_step(self, batch, edited_model, base_logits_vis, base_image_logits_vis, base_logits_tex):
        info_dict = {}

        ##############################################################################
        # ----------------------------Test: Visual Edit------------------------------#
        with torch.no_grad():
            # set lora: visual inference
            edited_model.model.set_adapter("visual")
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["visual_edit"]["edit_inner"])
            inner_batch_labels = batch["visual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels) # edit_loss_fn이 어딨지?
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(T-Gen)
            post_edit_outputs = edited_model(batch["visual_edit"]["edit_outer"])
            post_batch_labels = batch["visual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels) # edit_loss_fn -> vis, text 한번에 적용해도 되는가
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # image rephrase(I-Gen)
            post_image_edit_outputs = edited_model(batch["visual_edit"]["edit_outer_image"])
            post_image_batch_labels = batch["visual_edit"]["edit_outer_image"]["labels"]
            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
            else:
                post_image_edit_logits = post_image_edit_outputs

            if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)
            else:
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])
            del post_image_edit_outputs, post_image_edit_logits
            torch.cuda.empty_cache()

            # text loc(T-Loc)
            post_base_outputs = edited_model(batch["visual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_vis, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

            # image loc(I-Loc)
            post_image_base_outputs = edited_model(batch["visual_edit"]["loc_image"])
            if not isinstance(post_image_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
            else:
                post_image_base_logits = post_image_base_outputs
            post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
            base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits_vis, dim=-1), k=10, dim=-1).indices
            del post_image_base_outputs, post_image_base_logits
            torch.cuda.empty_cache()

        info_dict['vis/inner/acc'] = inner_edit_dict["acc"].item() # copy안해도 되는가? -> item은 int/float이 직접 반환됨. 따라서 ㅇㅇ
        info_dict['vis/edit/acc'] = post_edit_dict["acc"].item()
        info_dict['vis/image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["vis/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["vis/image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ##############################################################################
        # ----------------------------Test: Textual Edit------------------------------#
        with torch.no_grad():
            # set lora: visual inference
            edited_model.model.set_adapter("textual")
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["textual_edit"]["edit_inner"])
            inner_batch_labels = batch["textual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(Generality)
            post_edit_outputs = edited_model(batch["textual_edit"]["edit_outer"])
            post_batch_labels = batch["textual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # text loc(Locality)
            post_base_outputs = edited_model(batch["textual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_tex, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

        info_dict['text/inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['text/edit/acc'] = post_edit_dict["acc"].item()
        info_dict["text/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ################ portability #################
        # set lora: visual&textual inference
        #### Deprecated version(not using PEFT) ####
        # if "merge" in edited_model.model.peft_config:
        #     edited_model.model.delete_adapter("merge")

        # weights = [1.0, 1.0]
        # adapter_name = "merge"
        # edited_model.model.add_weighted_adapter(["visual", "textual"], weights, adapter_name, combination_type="linear") # linear -> cat, dar_linear 추천
        # edited_model.model.set_adapter("merge")

        # set lora: visual&textual inference
        edited_model.model.set_adapter(["textual","visual"])    

        if batch['port'] is not None:
            assert len(batch['port']) == 1, "batch['port'] exist and have only one element"
            port = batch['port'][0]
            with torch.no_grad():
                port_outputs = edited_model(port)
                port_labels = port["labels"]
                if not isinstance(port_outputs, torch.Tensor):
                    port_logits = port_outputs.logits
                else:
                    port_logits = port_outputs
                if port_logits.shape[1] > port_labels.shape[1]:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
                else:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
                port_acc = port_dict["acc"].item()
                del port_outputs, port_logits
                torch.cuda.empty_cache()

            info_dict['port/acc'] = port_acc
            ################ portability #################

        return info_dict

    # BLIP2
    def test_sequencial_rag_with_two_lora_parallel(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") # 궁금한게, 200개에 대한 batch
        averager = RunningStatAverager("val")

        start_time = time.time()
        ## 저장할 내용
        val_data_store = []

        # visul-data
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        # textual-data
        base_logits_store_tex = []

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 1. Inference Output for test locality(visual & textual 데이터/출력 저장 & 출력)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                # 1.1) visual edit part
                val_data_store.append(batch) # batch 데이터 저장
                with torch.no_grad():
                    base_outputs = self.model(batch["visual_edit"]["loc"]) # T-Loc inference 저장 # self.model -> ft
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["visual_edit"]["loc_image"]) # I-Loc inference 저장
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach())

                # 1.2) textual edit part
                with torch.no_grad():
                    base_outputs = self.model(batch["textual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_tex.append(base_logits.clone().detach())

                pbar.update(1)
            else:
                break
        pbar.close()

        ## 2. Model edit & Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            # 2.1) Model Edit (Update for a batch)
            # 2.1.1) Visual Edit(first)
            self.model.model.set_adapter("visual") # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["visual_edit"]["edit_inner"], mode = "visual" , peft = True)
            # 필요시, edit 후 바로 성능 체크

            # 2.1.2) Textual Edit(second) ★☆★☆★☆ --> __getitem__ 시 확인. collate_fn
            self.model.model.set_adapter("textual")  # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["textual_edit"]["edit_inner"], mode = "textual" , peft = True)

            # 2.2) Test with GAP
            if val_step >= gap_num: 
                # 기존 저장했던 batch, t-loc & i-loc-logit 불러옴. For Test
                stored_batch = val_data_store.pop(0) # vis + text
                stored_base_logits_vis = base_logits_store_vis.pop(0)
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0)
                stored_base_logits_tex = base_logits_store_tex.pop(0)

                # Test Sequential Edit(only inference & test) - vis / text 모두 다 평가해야 함.
                info_dict = self.test_sequencial_rag_with_two_lora_parallel_step(
                    stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_tex
                    )
                averager.add(info_dict)

            # logging?
            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log_CompositionalEdit( ## ★☆★ 수정 필요 ★☆★ ##
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Logging Results ## 
        if log:
            self._inline_seq_log_CompositionalEdit(val_step, averager.average(), start_time, steps) ## ★☆★ 수정 필요 ★☆★ ##
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

                
        if 'llava' in self.config.model_name.lower():
            result_dir = f"results/results_sequencial/composition/RAG_with_two_lora"
        elif 'blip2' in self.config.model_name.lower() :
            result_dir = f"results/results_sequencial/blip2/composition/RAG_with_two_lora"

        results_path = os.path.join(result_dir,f"{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json")

        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats  
    def test_sequencial_rag_with_two_lora_parallel_simple(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") # 궁금한게, 200개에 대한 batch
        averager = RunningStatAverager("val")

        start_time = time.time()
        ## 저장할 내용
        val_data_store = []

        # visul-data
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        # textual-data
        base_logits_store_tex = []

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 1. Inference Output for test locality(visual & textual 데이터/출력 저장 & 출력)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                # 1.1) visual edit part
                val_data_store.append(batch) # batch 데이터 저장
                with torch.no_grad():
                    base_outputs = self.model(batch["visual_edit"]["loc"]) # T-Loc inference 저장 # self.model -> ft
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["visual_edit"]["loc_image"]) # I-Loc inference 저장
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach())

                # 1.2) textual edit part
                with torch.no_grad():
                    base_outputs = self.model(batch["textual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_tex.append(base_logits.clone().detach())

                pbar.update(1)
            else:
                break
        pbar.close()

        ## 2. Model edit & Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            # 2.1) Model Edit (Update for a batch)
            # 2.1.1) Visual Edit(first)
            self.model.model.set_adapter("visual") # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["visual_edit"]["edit_inner"], mode = "visual" , peft = True)
            # 필요시, edit 후 바로 성능 체크

            # 2.1.2) Textual Edit(second) ★☆★☆★☆ --> __getitem__ 시 확인. collate_fn
            self.model.model.set_adapter("textual")  # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["textual_edit"]["edit_inner"], mode = "textual" , peft = True)

            # 2.2) Test with GAP
            if val_step >= gap_num: 
                # 기존 저장했던 batch, t-loc & i-loc-logit 불러옴. For Test
                stored_batch = val_data_store.pop(0) # vis + text
                stored_base_logits_vis = base_logits_store_vis.pop(0)
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0)
                stored_base_logits_tex = base_logits_store_tex.pop(0)

                # Test Sequential Edit(only inference & test) - vis / text 모두 다 평가해야 함.
                info_dict = self.test_sequencial_rag_with_two_lora_parallel_step(
                    stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_tex
                    )
                averager.add(info_dict)

            # logging?
            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log_CompositionalEdit( ## ★☆★ 수정 필요 ★☆★ ##
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Logging Results ## 
        if log:
            self._inline_seq_log_CompositionalEdit(val_step, averager.average(), start_time, steps) ## ★☆★ 수정 필요 ★☆★ ##
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

                
        if 'llava' in self.config.model_name.lower():
            result_dir = f"results/results_sequencial/composition/RAG_with_two_lora_simple"
        elif 'blip2' in self.config.model_name.lower() :
            result_dir = f"results/results_sequencial/blip2/composition/RAG_with_two_lora_simple"

        results_path = os.path.join(result_dir,f"{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json")

        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats  
    
    def test_sequencial_rag_with_two_lora_parallel_step(self, batch, edited_model, base_logits_vis, base_image_logits_vis, base_logits_tex):
        info_dict = {}

        ##############################################################################
        # ----------------------------Test: Visual Edit------------------------------#
        with torch.no_grad():
            # set lora: visual inference
            edited_model.model.set_adapter("visual")
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["visual_edit"]["edit_inner"])
            inner_batch_labels = batch["visual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels) # edit_loss_fn이 어딨지?
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(T-Gen)
            post_edit_outputs = edited_model(batch["visual_edit"]["edit_outer"])
            post_batch_labels = batch["visual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels) # edit_loss_fn -> vis, text 한번에 적용해도 되는가
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # image rephrase(I-Gen)
            post_image_edit_outputs = edited_model(batch["visual_edit"]["edit_outer_image"])
            post_image_batch_labels = batch["visual_edit"]["edit_outer_image"]["labels"]
            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
            else:
                post_image_edit_logits = post_image_edit_outputs

            if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)
            else:
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])
            del post_image_edit_outputs, post_image_edit_logits
            torch.cuda.empty_cache()

            # text loc(T-Loc)
            post_base_outputs = edited_model(batch["visual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_vis, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

            # image loc(I-Loc)
            post_image_base_outputs = edited_model(batch["visual_edit"]["loc_image"])
            if not isinstance(post_image_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
            else:
                post_image_base_logits = post_image_base_outputs
            post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
            base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits_vis, dim=-1), k=10, dim=-1).indices
            del post_image_base_outputs, post_image_base_logits
            torch.cuda.empty_cache()

        info_dict['vis/inner/acc'] = inner_edit_dict["acc"].item() # copy안해도 되는가? -> item은 int/float이 직접 반환됨. 따라서 ㅇㅇ
        info_dict['vis/edit/acc'] = post_edit_dict["acc"].item()
        info_dict['vis/image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["vis/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["vis/image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ##############################################################################
        # ----------------------------Test: Textual Edit------------------------------#
        with torch.no_grad():
            # set lora: visual inference
            edited_model.model.set_adapter("textual")
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["textual_edit"]["edit_inner"])
            inner_batch_labels = batch["textual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(Generality)
            post_edit_outputs = edited_model(batch["textual_edit"]["edit_outer"])
            post_batch_labels = batch["textual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # text loc(Locality)
            post_base_outputs = edited_model(batch["textual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_tex, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

        info_dict['text/inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['text/edit/acc'] = post_edit_dict["acc"].item()
        info_dict["text/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ################ portability #################
        # set lora: visual&textual inference
        edited_model.model.set_adapter(["textual","visual"])

        if batch['port'] is not None:
            assert len(batch['port']) == 1, "batch['port'] exist and have only one element"
            port = batch['port'][0]
            with torch.no_grad():
                port_outputs = edited_model(port)
                port_labels = port["labels"]
                if not isinstance(port_outputs, torch.Tensor):
                    port_logits = port_outputs.logits
                else:
                    port_logits = port_outputs
                if port_logits.shape[1] > port_labels.shape[1]:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
                else:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
                port_acc = port_dict["acc"].item()
                del port_outputs, port_logits
                torch.cuda.empty_cache()

            info_dict['port/acc'] = port_acc
            ################ portability #################

        return info_dict


    ## TEST - RAG + LoRA -> compositonal edit2(mix 방법 변경)
    def test_sequencial_rag_with_two_lora_connector(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") # 궁금한게, 200개에 대한 batch
        averager = RunningStatAverager("val")

        start_time = time.time()
        ## 저장할 내용
        val_data_store = []

        # visul-data
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        # textual-data
        base_logits_store_tex = []

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 1. Inference Output for test locality(visual & textual 데이터/출력 저장 & 출력)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                # 1.1) visual edit part
                val_data_store.append(batch) # batch 데이터 저장
                with torch.no_grad():
                    base_outputs = self.model(batch["visual_edit"]["loc"]) # T-Loc inference 저장 # self.model -> ft
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["visual_edit"]["loc_image"]) # I-Loc inference 저장
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach())

                # 1.2) textual edit part
                with torch.no_grad():
                    base_outputs = self.model(batch["textual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_tex.append(base_logits.clone().detach())

                pbar.update(1)
            else:
                break
        pbar.close()

        ## 2. Model edit & Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            # 2.1) Model Edit (Update for a batch)
            # 2.1.1) Visual Edit(first)
            self.model.model.set_adapter("visual") # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["visual_edit"]["edit_inner"], mode = "visual" , peft = True)
            # 필요시, edit 후 바로 성능 체크

            # 2.1.2) Textual Edit(second) ★☆★☆★☆ --> __getitem__ 시 확인. collate_fn
            self.model.model.set_adapter("textual")  # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["textual_edit"]["edit_inner"], mode = "textual" , peft = True)

            # 2.2) Test with GAP
            if val_step >= gap_num: 
                # 기존 저장했던 batch, t-loc & i-loc-logit 불러옴. For Test
                stored_batch = val_data_store.pop(0) # vis + text
                stored_base_logits_vis = base_logits_store_vis.pop(0)
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0)
                stored_base_logits_tex = base_logits_store_tex.pop(0)

                # Test Sequential Edit(only inference & test) - vis / text 모두 다 평가해야 함.
                info_dict = self.test_sequencial_compositional_two_step(
                    stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_tex
                    )
                averager.add(info_dict)

            # logging?
            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log_CompositionalEdit( ## ★☆★ 수정 필요 ★☆★ ##
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Logging Results ## 
        if log:
            self._inline_seq_log_CompositionalEdit(val_step, averager.average(), start_time, steps) ## ★☆★ 수정 필요 ★☆★ ##
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        results_path = f"results/results_sequencial/composition/RAG_with_two_lora_connector/{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json"
        
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats
    

        ## TEST - compositonal - two
    def test_sequencial_rag_with_two_lora_connector_step(self, batch, edited_model, base_logits_vis, base_image_logits_vis, base_logits_tex):
        info_dict = {}

        ##############################################################################
        # ----------------------------Test: Visual Edit------------------------------#
        with torch.no_grad():
            # set lora: visual inference
            edited_model.model.set_adapter("visual")
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["visual_edit"]["edit_inner"])
            inner_batch_labels = batch["visual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels) # edit_loss_fn이 어딨지?
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(T-Gen)
            post_edit_outputs = edited_model(batch["visual_edit"]["edit_outer"])
            post_batch_labels = batch["visual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels) # edit_loss_fn -> vis, text 한번에 적용해도 되는가
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # image rephrase(I-Gen)
            post_image_edit_outputs = edited_model(batch["visual_edit"]["edit_outer_image"])
            post_image_batch_labels = batch["visual_edit"]["edit_outer_image"]["labels"]
            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
            else:
                post_image_edit_logits = post_image_edit_outputs

            if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)
            else:
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])
            del post_image_edit_outputs, post_image_edit_logits
            torch.cuda.empty_cache()

            # text loc(T-Loc)
            post_base_outputs = edited_model(batch["visual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_vis, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

            # image loc(I-Loc)
            post_image_base_outputs = edited_model(batch["visual_edit"]["loc_image"])
            if not isinstance(post_image_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
            else:
                post_image_base_logits = post_image_base_outputs
            post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
            base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits_vis, dim=-1), k=10, dim=-1).indices
            del post_image_base_outputs, post_image_base_logits
            torch.cuda.empty_cache()

        info_dict['vis/inner/acc'] = inner_edit_dict["acc"].item() # copy안해도 되는가? -> item은 int/float이 직접 반환됨. 따라서 ㅇㅇ
        info_dict['vis/edit/acc'] = post_edit_dict["acc"].item()
        info_dict['vis/image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["vis/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["vis/image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ##############################################################################
        # ----------------------------Test: Textual Edit------------------------------#
        with torch.no_grad():
            # set lora: visual inference
            edited_model.model.set_adapter("textual")
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["textual_edit"]["edit_inner"])
            inner_batch_labels = batch["textual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(Generality)
            post_edit_outputs = edited_model(batch["textual_edit"]["edit_outer"])
            post_batch_labels = batch["textual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # text loc(Locality)
            post_base_outputs = edited_model(batch["textual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_tex, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

        info_dict['text/inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['text/edit/acc'] = post_edit_dict["acc"].item()
        info_dict["text/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ################ portability #################
        # set lora: visual&textual inference
        # 기존 어댑터가 존재하면 삭제
        if "merge" in edited_model.model.peft_config:
            edited_model.model.delete_adapter("merge")
        
        weights = [1.0, 1.0]
        adapter_name = "merge"
        edited_model.model.add_weighted_adapter(["visual","textual"], weights, adapter_name, combination_type="dare_linear") # linear-> 지식충돌? , ties => 멀티모달 조합에 적합, dare_linea(46% -> 별로)
        edited_model.model.set_adapter("merge")

        if batch['port'] is not None:
            assert len(batch['port']) == 1, "batch['port'] exist and have only one element"
            port = batch['port'][0]
            with torch.no_grad():
                port_outputs = edited_model(port)
                port_labels = port["labels"]
                if not isinstance(port_outputs, torch.Tensor):
                    port_logits = port_outputs.logits
                else:
                    port_logits = port_outputs
                if port_logits.shape[1] > port_labels.shape[1]:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
                else:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
                port_acc = port_dict["acc"].item()
                del port_outputs, port_logits
                torch.cuda.empty_cache()

            info_dict['port/acc'] = port_acc
            ################ portability #################

        return info_dict

    ## TEST - RAG + LoRA -> compositonal edit3(mix 방법 변경)
    def test_sequencial_rag_with_two_lora_wo_prompt(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") # 궁금한게, 200개에 대한 batch
        averager = RunningStatAverager("val")

        start_time = time.time()
        ## 저장할 내용
        val_data_store = []

        # visul-data
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        # textual-data
        base_logits_store_tex = []

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 1. Inference Output for test locality(visual & textual 데이터/출력 저장 & 출력)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                # 1.1) visual edit part
                val_data_store.append(batch) # batch 데이터 저장
                with torch.no_grad():
                    base_outputs = self.model(batch["visual_edit"]["loc"]) # T-Loc inference 저장 # self.model -> ft
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["visual_edit"]["loc_image"]) # I-Loc inference 저장
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach())

                # 1.2) textual edit part
                with torch.no_grad():
                    base_outputs = self.model(batch["textual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_tex.append(base_logits.clone().detach())

                pbar.update(1)
            else:
                break
        pbar.close()

        ## 2. Model edit & Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            # 2.1) Model Edit (Update for a batch)
            # 2.1.1) Visual Edit(first)
            self.model.model.set_adapter("visual") # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["visual_edit"]["edit_inner"], mode = "visual" , peft = True)
            # 필요시, edit 후 바로 성능 체크

            # 2.1.2) Textual Edit(second) ★☆★☆★☆ --> __getitem__ 시 확인. collate_fn
            self.model.model.set_adapter("textual")  # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["textual_edit"]["edit_inner"], mode = "textual" , peft = True)

            # 2.2) Test with GAP
            if val_step >= gap_num: 
                # 기존 저장했던 batch, t-loc & i-loc-logit 불러옴. For Test
                stored_batch = val_data_store.pop(0) # vis + text
                stored_base_logits_vis = base_logits_store_vis.pop(0)
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0)
                stored_base_logits_tex = base_logits_store_tex.pop(0)

                # Test Sequential Edit(only inference & test) - vis / text 모두 다 평가해야 함.
                info_dict = self.test_sequencial_compositional_two_step(
                    stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_tex
                    )
                averager.add(info_dict)

            # logging?
            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log_CompositionalEdit( ## ★☆★ 수정 필요 ★☆★ ##
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Logging Results ## 
        if log:
            self._inline_seq_log_CompositionalEdit(val_step, averager.average(), start_time, steps) ## ★☆★ 수정 필요 ★☆★ ##
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        results_path = f"results/results_sequencial/composition/RAG_with_two_lora_wo_prompt/{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json"
        
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats
    

        ## TEST - compositonal - two
    def test_sequencial_rag_with_two_lora_wo_prompt_step(self, batch, edited_model, base_logits_vis, base_image_logits_vis, base_logits_tex):
        info_dict = {}

        ##############################################################################
        # ----------------------------Test: Visual Edit------------------------------#
        with torch.no_grad():
            # set lora: visual inference
            edited_model.model.set_adapter("visual")
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["visual_edit"]["edit_inner"])
            inner_batch_labels = batch["visual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels) # edit_loss_fn이 어딨지?
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(T-Gen)
            post_edit_outputs = edited_model(batch["visual_edit"]["edit_outer"])
            post_batch_labels = batch["visual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels) # edit_loss_fn -> vis, text 한번에 적용해도 되는가
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # image rephrase(I-Gen)
            post_image_edit_outputs = edited_model(batch["visual_edit"]["edit_outer_image"])
            post_image_batch_labels = batch["visual_edit"]["edit_outer_image"]["labels"]
            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
            else:
                post_image_edit_logits = post_image_edit_outputs

            if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)
            else:
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])
            del post_image_edit_outputs, post_image_edit_logits
            torch.cuda.empty_cache()

            # text loc(T-Loc)
            post_base_outputs = edited_model(batch["visual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_vis, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

            # image loc(I-Loc)
            post_image_base_outputs = edited_model(batch["visual_edit"]["loc_image"])
            if not isinstance(post_image_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
            else:
                post_image_base_logits = post_image_base_outputs
            post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
            base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits_vis, dim=-1), k=10, dim=-1).indices
            del post_image_base_outputs, post_image_base_logits
            torch.cuda.empty_cache()

        info_dict['vis/inner/acc'] = inner_edit_dict["acc"].item() # copy안해도 되는가? -> item은 int/float이 직접 반환됨. 따라서 ㅇㅇ
        info_dict['vis/edit/acc'] = post_edit_dict["acc"].item()
        info_dict['vis/image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["vis/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["vis/image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ##############################################################################
        # ----------------------------Test: Textual Edit------------------------------#
        with torch.no_grad():
            # set lora: visual inference
            edited_model.model.set_adapter("textual")
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["textual_edit"]["edit_inner"])
            inner_batch_labels = batch["textual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(Generality)
            post_edit_outputs = edited_model(batch["textual_edit"]["edit_outer"])
            post_batch_labels = batch["textual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # text loc(Locality)
            post_base_outputs = edited_model(batch["textual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_tex, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

        info_dict['text/inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['text/edit/acc'] = post_edit_dict["acc"].item()
        info_dict["text/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ################ portability #################
        # set lora: visual&textual inference
        # 기존 어댑터가 존재하면 삭제
        if "merge" in edited_model.model.peft_config:
            edited_model.model.delete_adapter("merge")

        weights = [1.0, 1.0]
        adapter_name = "merge"
        edited_model.model.add_weighted_adapter(["visual","textual"], weights, adapter_name, combination_type="cat") # linear-> 지식충돌? , ties => 멀티모달 조합에 적합    
        edited_model.model.set_adapter("merge")

        if batch['port'] is not None:
            assert len(batch['port']) == 1, "batch['port'] exist and have only one element"
            port = batch['port'][0]
            with torch.no_grad():
                port_outputs = edited_model(port)
                port_labels = port["labels"]
                if not isinstance(port_outputs, torch.Tensor):
                    port_logits = port_outputs.logits
                else:
                    port_logits = port_outputs
                if port_logits.shape[1] > port_labels.shape[1]:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
                else:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
                port_acc = port_dict["acc"].item()
                del port_outputs, port_logits
                torch.cuda.empty_cache()

            info_dict['port/acc'] = port_acc
            ################ portability #################

        return info_dict

    
    #### ★★★ 여기부터 쭉 보기(주요 실험들) ★★★ #### 
    ###############################################################
    ### --- TEST - compositonal - two lora + Connector(att) --- ###
    def test_sequencial_compositional_connector_ffn(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") # 궁금한게, 200개에 대한 batch
        averager = RunningStatAverager("val")

        start_time = time.time()
        ## 저장할 내용
        val_data_store = []

        # visul-data
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        # textual-data
        base_logits_store_tex = []

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 1. Inference Output for test locality(visual & textual 데이터/출력 저장 & 출력)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                # 1.1) visual edit part
                val_data_store.append(batch) # batch 데이터 저장
                with torch.no_grad():
                    base_outputs = self.model(batch["visual_edit"]["loc"]) # T-Loc inference 저장 # self.model -> ft
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["visual_edit"]["loc_image"]) # I-Loc inference 저장
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach())

                # 1.2) textual edit part
                with torch.no_grad():
                    base_outputs = self.model(batch["textual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_tex.append(base_logits.clone().detach())

                pbar.update(1)
            else:
                break
        pbar.close()

        ## 2. Model edit & Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            # 2.1) Model Edit (Update for a batch)
            # 2.1.1) Visual Edit(first)
            self.model.model.set_adapter("visual") # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["visual_edit"]["edit_inner"], mode = "visual" , peft = True)

            # 2.1.2) Textual Edit(second) 
            self.model.model.set_adapter("textual")  # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["textual_edit"]["edit_inner"], mode = "textual" , peft = True)

            # 2.1.3) Compositional Edit(second) ★ mlp 학습 o
            if val_step > 5:
                edited_model.model.set_adapter(["textual","visual","connector"])
                edited_model, _ = edited_model.edit(batch["port"][0], connector_mode=True) # cond? 이거 안되나

            # 2.2) Test with GAP
            if val_step >= gap_num: 
                # 기존 저장했던 batch, t-loc & i-loc-logit 불러옴. For Test
                stored_batch = val_data_store.pop(0) # vis + text
                stored_base_logits_vis = base_logits_store_vis.pop(0)
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0)
                stored_base_logits_tex = base_logits_store_tex.pop(0)

                # Test Sequential Edit(only inference & test) - vis / text 모두 다 평가해야 함.
                info_dict = self.test_sequencial_compositional_connector_ffn_step(
                    stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_tex
                    )
                averager.add(info_dict)

            # logging?
            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log_CompositionalEdit( ## ★☆★ 수정 필요 ★☆★ ##
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Logging Results ## 
        if log:
            self._inline_seq_log_CompositionalEdit(val_step, averager.average(), start_time, steps) ## ★☆★ 수정 필요 ★☆★ ##
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        results_path = f"results/results_sequencial/composition/two_lora_connect_ffn/new/{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json"
        
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        if gap_num == 0:
            try: # lora weight 저장
                from peft import LoraConfig, TaskType, get_peft_model, PeftConfig, PeftModel
                connector_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=8,
                        lora_alpha=16,
                        lora_dropout=0.05,
                        target_modules=["q_proj", "k_proj"]
                    )
                peft_model = get_peft_model(self.model.model.base_model.model, connector_config)
                peft_model.delete_adapter("default")
                peft_model = peft_model.cpu()
                peft_model.save_pretrained("results/results_sequencial/composition/two_lora_connect_ffn/new")
                # 저장 후 메모리 해제
                del peft_model

                torch.cuda.empty_cache()
                print("LoRA + (gap0, train_composition.json) 모델 저장 완료 -> \"results/results_sequencial/composition/two_lora_connect_ffn/new\" ")
            except:
                print("LoRA, MLP 모델 저장 실패")

        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats
    

        ## TEST - compositonal - two
    def test_sequencial_compositional_connector_ffn_step(self, batch, edited_model, base_logits_vis, base_image_logits_vis, base_logits_tex):
        info_dict = {}

        ##############################################################################
        # ----------------------------Test: Visual Edit------------------------------#
        with torch.no_grad():
            # set lora: visual inference
            edited_model.model.set_adapter("visual")
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["visual_edit"]["edit_inner"])
            inner_batch_labels = batch["visual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels) # edit_loss_fn이 어딨지?
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(T-Gen)
            post_edit_outputs = edited_model(batch["visual_edit"]["edit_outer"])
            post_batch_labels = batch["visual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels) # edit_loss_fn -> vis, text 한번에 적용해도 되는가
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # image rephrase(I-Gen)
            post_image_edit_outputs = edited_model(batch["visual_edit"]["edit_outer_image"])
            post_image_batch_labels = batch["visual_edit"]["edit_outer_image"]["labels"]
            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
            else:
                post_image_edit_logits = post_image_edit_outputs

            if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)
            else:
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])
            del post_image_edit_outputs, post_image_edit_logits
            torch.cuda.empty_cache()

            # text loc(T-Loc)
            post_base_outputs = edited_model(batch["visual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_vis, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

            # image loc(I-Loc)
            post_image_base_outputs = edited_model(batch["visual_edit"]["loc_image"])
            if not isinstance(post_image_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
            else:
                post_image_base_logits = post_image_base_outputs
            post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
            base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits_vis, dim=-1), k=10, dim=-1).indices
            del post_image_base_outputs, post_image_base_logits
            torch.cuda.empty_cache()

        info_dict['vis/inner/acc'] = inner_edit_dict["acc"].item() # copy안해도 되는가? -> item은 int/float이 직접 반환됨. 따라서 ㅇㅇ
        info_dict['vis/edit/acc'] = post_edit_dict["acc"].item()
        info_dict['vis/image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["vis/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["vis/image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ##############################################################################
        # ----------------------------Test: Textual Edit------------------------------#
        with torch.no_grad():
            # set lora: visual inference
            edited_model.model.set_adapter("textual")
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["textual_edit"]["edit_inner"])
            inner_batch_labels = batch["textual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(Generality)
            post_edit_outputs = edited_model(batch["textual_edit"]["edit_outer"])
            post_batch_labels = batch["textual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # text loc(Locality)
            post_base_outputs = edited_model(batch["textual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_tex, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

        info_dict['text/inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['text/edit/acc'] = post_edit_dict["acc"].item()
        info_dict["text/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ################ portability #################
        # set lora: visual&textual inference
        edited_model.model.set_adapter(["textual","visual","connector"])

        if batch['port'] is not None:
            assert len(batch['port']) == 1, "batch['port'] exist and have only one element"
            port = batch['port'][0]
            with torch.no_grad():
                port_outputs = edited_model(port)
                port_labels = port["labels"]
                if not isinstance(port_outputs, torch.Tensor):
                    port_logits = port_outputs.logits
                else:
                    port_logits = port_outputs
                if port_logits.shape[1] > port_labels.shape[1]:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
                else:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
                port_acc = port_dict["acc"].item()
                del port_outputs, port_logits
                torch.cuda.empty_cache()

            info_dict['port/acc'] = port_acc
            ################ portability #################

        return info_dict

    # TEST - compositonal - two lora + Connector(ffn) + rag
    def test_sequencial_compositional_connector_ffn_rag(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") # 궁금한게, 200개에 대한 batch
        averager = RunningStatAverager("val")

        start_time = time.time()
        ## 저장할 내용
        val_data_store = []

        # visul-data
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        # textual-data
        base_logits_store_tex = []

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 1. Inference Output for test locality(visual & textual 데이터/출력 저장 & 출력)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                # 1.1) visual edit part
                val_data_store.append(batch) # batch 데이터 저장
                with torch.no_grad():
                    base_outputs = self.model(batch["visual_edit"]["loc"]) # T-Loc inference 저장 # self.model -> ft
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["visual_edit"]["loc_image"]) # I-Loc inference 저장
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach())

                # 1.2) textual edit part
                with torch.no_grad():
                    base_outputs = self.model(batch["textual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_tex.append(base_logits.clone().detach())

                pbar.update(1)
            else:
                break
        pbar.close()

        ## 2. Model edit & Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            # 2.1) Model Edit (Update for a batch)
            # 2.1.1) Visual Edit(first)
            self.model.model.set_adapter("visual") # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["visual_edit"]["edit_inner"], mode = "visual" , peft = True)

            # 2.1.2) Textual Edit(second) 
            self.model.model.set_adapter("textual")  # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["textual_edit"]["edit_inner"], mode = "textual" , peft = True)

            # 2.1.3) Compositional Edit(second) ★ mlp 학습 o
            if val_step > 5:
                edited_model.model.set_adapter(["textual","visual","connector"])
                edited_model, _ = edited_model.edit(batch["port"][0], connector_mode=True) # cond? 이거 안되나

            # 2.2) Test with GAP
            if val_step >= gap_num: 
                # 기존 저장했던 batch, t-loc & i-loc-logit 불러옴. For Test
                stored_batch = val_data_store.pop(0) # vis + text
                stored_base_logits_vis = base_logits_store_vis.pop(0)
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0)
                stored_base_logits_tex = base_logits_store_tex.pop(0)

                # Test Sequential Edit(only inference & test) - vis / text 모두 다 평가해야 함.
                info_dict = self.test_sequencial_compositional_connector_ffn_rag_step(
                    stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_tex
                    )
                averager.add(info_dict)

            # logging?
            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log_CompositionalEdit( ## ★☆★ 수정 필요 ★☆★ ##
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Logging Results ## 
        if log:
            self._inline_seq_log_CompositionalEdit(val_step, averager.average(), start_time, steps) ## ★☆★ 수정 필요 ★☆★ ##
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        results_path = f"results/results_sequencial/composition/two_lora_connect_ffn_rag/{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json"
        
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        if gap_num == 0:
            try: # lora weight 저장
                from peft import LoraConfig, TaskType, get_peft_model, PeftConfig, PeftModel
                connector_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=8,
                        lora_alpha=16,
                        lora_dropout=0.05,
                        target_modules=["q_proj", "k_proj"]
                    )
                peft_model = get_peft_model(self.model.model.base_model.model, connector_config)
                peft_model.delete_adapter("default")
                peft_model = peft_model.cpu()
                peft_model.save_pretrained("results/results_sequencial/composition/two_lora_connect_ffn_rag")
                # 저장 후 메모리 해제
                del peft_model

                torch.cuda.empty_cache()
                print("LoRA + (gap0, train_composition.json) 모델 저장 완료 -> \"results/results_sequencial/composition/two_lora_connect_ffn_rag\" ")
            except:
                print("LoRA, MLP 모델 저장 실패")

        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats
    

        ## TEST - compositonal - two
    def test_sequencial_compositional_connector_ffn_rag_70(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") # 궁금한게, 200개에 대한 batch
        averager = RunningStatAverager("val")

        start_time = time.time()
        ## 저장할 내용
        val_data_store = []

        # visul-data
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        # textual-data
        base_logits_store_tex = []

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 1. Inference Output for test locality(visual & textual 데이터/출력 저장 & 출력)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                # 1.1) visual edit part
                val_data_store.append(batch) # batch 데이터 저장
                with torch.no_grad():
                    base_outputs = self.model(batch["visual_edit"]["loc"]) # T-Loc inference 저장 # self.model -> ft
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["visual_edit"]["loc_image"]) # I-Loc inference 저장
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach())

                # 1.2) textual edit part
                with torch.no_grad():
                    base_outputs = self.model(batch["textual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_tex.append(base_logits.clone().detach())

                pbar.update(1)
            else:
                break
        pbar.close()

        ## 2. Model edit & Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            # 2.1) Model Edit (Update for a batch)
            # 2.1.1) Visual Edit(first)
            self.model.model.set_adapter("visual") # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["visual_edit"]["edit_inner"], mode = "visual" , peft = True)

            # 2.1.2) Textual Edit(second) 
            self.model.model.set_adapter("textual")  # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["textual_edit"]["edit_inner"], mode = "textual" , peft = True)

            # 2.1.3) Compositional Edit(second) ★ mlp 학습 o
            if val_step > 5:
                edited_model.model.set_adapter(["textual","visual","connector"])
                edited_model, _ = edited_model.edit(batch["port"][0], connector_mode=True) # cond? 이거 안되나

            # 2.2) Test with GAP
            if val_step >= gap_num: 
                # 기존 저장했던 batch, t-loc & i-loc-logit 불러옴. For Test
                stored_batch = val_data_store.pop(0) # vis + text
                stored_base_logits_vis = base_logits_store_vis.pop(0)
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0)
                stored_base_logits_tex = base_logits_store_tex.pop(0)

                # Test Sequential Edit(only inference & test) - vis / text 모두 다 평가해야 함.
                info_dict = self.test_sequencial_compositional_connector_ffn_rag_step(
                    stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_tex
                    )
                averager.add(info_dict)

            # logging?
            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log_CompositionalEdit( ## ★☆★ 수정 필요 ★☆★ ##
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Logging Results ## 
        if log:
            self._inline_seq_log_CompositionalEdit(val_step, averager.average(), start_time, steps) ## ★☆★ 수정 필요 ★☆★ ##
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        results_path = f"results/results_sequencial/composition/two_lora_connect_ffn_rag_70/{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json"
        
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        if gap_num == 0:
            try: # lora weight 저장
                from peft import LoraConfig, TaskType, get_peft_model, PeftConfig, PeftModel
                connector_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=8,
                        lora_alpha=16,
                        lora_dropout=0.05,
                        target_modules=["q_proj", "k_proj"]
                    )
                peft_model = get_peft_model(self.model.model.base_model.model, connector_config)
                peft_model.delete_adapter("default")
                peft_model = peft_model.cpu()
                peft_model.save_pretrained("results/results_sequencial/composition/two_lora_connect_ffn_rag_70")
                # 저장 후 메모리 해제
                del peft_model

                torch.cuda.empty_cache()
                print("LoRA + (gap0, train_composition.json) 모델 저장 완료 -> \"results/results_sequencial/composition/two_lora_connect_ffn_rag_70\" ")
            except:
                print("LoRA, MLP 모델 저장 실패")

        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats
    

        ## TEST - compositonal - two
    def test_sequencial_compositional_connector_ffn_rag_50(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") # 궁금한게, 200개에 대한 batch
        averager = RunningStatAverager("val")

        start_time = time.time()
        ## 저장할 내용
        val_data_store = []

        # visul-data
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        # textual-data
        base_logits_store_tex = []

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 1. Inference Output for test locality(visual & textual 데이터/출력 저장 & 출력)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                # 1.1) visual edit part
                val_data_store.append(batch) # batch 데이터 저장
                with torch.no_grad():
                    base_outputs = self.model(batch["visual_edit"]["loc"]) # T-Loc inference 저장 # self.model -> ft
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["visual_edit"]["loc_image"]) # I-Loc inference 저장
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach())

                # 1.2) textual edit part
                with torch.no_grad():
                    base_outputs = self.model(batch["textual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_tex.append(base_logits.clone().detach())

                pbar.update(1)
            else:
                break
        pbar.close()

        ## 2. Model edit & Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            # 2.1) Model Edit (Update for a batch)
            # 2.1.1) Visual Edit(first)
            self.model.model.set_adapter("visual") # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["visual_edit"]["edit_inner"], mode = "visual" , peft = True)

            # 2.1.2) Textual Edit(second) 
            self.model.model.set_adapter("textual")  # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["textual_edit"]["edit_inner"], mode = "textual" , peft = True)

            # 2.1.3) Compositional Edit(second) ★ mlp 학습 o
            if val_step > 5:
                edited_model.model.set_adapter(["textual","visual","connector"])
                edited_model, _ = edited_model.edit(batch["port"][0], connector_mode=True) # cond? 이거 안되나

            # 2.2) Test with GAP
            if val_step >= gap_num: 
                # 기존 저장했던 batch, t-loc & i-loc-logit 불러옴. For Test
                stored_batch = val_data_store.pop(0) # vis + text
                stored_base_logits_vis = base_logits_store_vis.pop(0)
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0)
                stored_base_logits_tex = base_logits_store_tex.pop(0)

                # Test Sequential Edit(only inference & test) - vis / text 모두 다 평가해야 함.
                info_dict = self.test_sequencial_compositional_connector_ffn_rag_step(
                    stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_tex
                    )
                averager.add(info_dict)

            # logging?
            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log_CompositionalEdit( ## ★☆★ 수정 필요 ★☆★ ##
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Logging Results ## 
        if log:
            self._inline_seq_log_CompositionalEdit(val_step, averager.average(), start_time, steps) ## ★☆★ 수정 필요 ★☆★ ##
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        results_path = f"results/results_sequencial/composition/two_lora_connect_ffn_rag_50/{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json"
        
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        if gap_num == 0:
            try: # lora weight 저장
                from peft import LoraConfig, TaskType, get_peft_model, PeftConfig, PeftModel
                connector_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=8,
                        lora_alpha=16,
                        lora_dropout=0.05,
                        target_modules=["q_proj", "k_proj"]
                    )
                peft_model = get_peft_model(self.model.model.base_model.model, connector_config)
                peft_model.delete_adapter("default")
                peft_model = peft_model.cpu()
                peft_model.save_pretrained("results/results_sequencial/composition/two_lora_connect_ffn_rag_50")
                # 저장 후 메모리 해제
                del peft_model

                torch.cuda.empty_cache()
                print("LoRA + (gap0, train_composition.json) 모델 저장 완료 -> \"results/results_sequencial/composition/two_lora_connect_ffn_rag_50\" ")
            except:
                print("LoRA, MLP 모델 저장 실패")

        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats
    

        ## TEST - compositonal - two


    def test_sequencial_compositional_connector_ffn_rag_step(self, batch, edited_model, base_logits_vis, base_image_logits_vis, base_logits_tex):
        info_dict = {}

        ##############################################################################
        # ----------------------------Test: Visual Edit------------------------------#
        with torch.no_grad():
            # set lora: visual inference
            edited_model.model.set_adapter("visual")
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["visual_edit"]["edit_inner"])
            inner_batch_labels = batch["visual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels) # edit_loss_fn이 어딨지?
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(T-Gen)
            post_edit_outputs = edited_model(batch["visual_edit"]["edit_outer"])
            post_batch_labels = batch["visual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels) # edit_loss_fn -> vis, text 한번에 적용해도 되는가
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # image rephrase(I-Gen)
            post_image_edit_outputs = edited_model(batch["visual_edit"]["edit_outer_image"])
            post_image_batch_labels = batch["visual_edit"]["edit_outer_image"]["labels"]
            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
            else:
                post_image_edit_logits = post_image_edit_outputs

            if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)
            else:
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])
            del post_image_edit_outputs, post_image_edit_logits
            torch.cuda.empty_cache()

            # text loc(T-Loc)
            post_base_outputs = edited_model(batch["visual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_vis, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

            # image loc(I-Loc)
            post_image_base_outputs = edited_model(batch["visual_edit"]["loc_image"])
            if not isinstance(post_image_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
            else:
                post_image_base_logits = post_image_base_outputs
            post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
            base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits_vis, dim=-1), k=10, dim=-1).indices
            del post_image_base_outputs, post_image_base_logits
            torch.cuda.empty_cache()

        info_dict['vis/inner/acc'] = inner_edit_dict["acc"].item() # copy안해도 되는가? -> item은 int/float이 직접 반환됨. 따라서 ㅇㅇ
        info_dict['vis/edit/acc'] = post_edit_dict["acc"].item()
        info_dict['vis/image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["vis/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["vis/image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ##############################################################################
        # ----------------------------Test: Textual Edit------------------------------#
        with torch.no_grad():
            # set lora: visual inference
            edited_model.model.set_adapter("textual")
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["textual_edit"]["edit_inner"])
            inner_batch_labels = batch["textual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(Generality)
            post_edit_outputs = edited_model(batch["textual_edit"]["edit_outer"])
            post_batch_labels = batch["textual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # text loc(Locality)
            post_base_outputs = edited_model(batch["textual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_tex, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

        info_dict['text/inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['text/edit/acc'] = post_edit_dict["acc"].item()
        info_dict["text/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ################ portability #################
        # set lora: visual&textual inference
        edited_model.model.set_adapter(["textual","visual","connector"])

        if batch['port'] is not None:
            assert len(batch['port']) == 1, "batch['port'] exist and have only one element"
            port = batch['port'][0]
            with torch.no_grad():
                port_outputs = edited_model(port)
                port_labels = port["labels"]
                if not isinstance(port_outputs, torch.Tensor):
                    port_logits = port_outputs.logits
                else:
                    port_logits = port_outputs
                if port_logits.shape[1] > port_labels.shape[1]:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
                else:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
                port_acc = port_dict["acc"].item()
                del port_outputs, port_logits
                torch.cuda.empty_cache()

            info_dict['port/acc'] = port_acc
            ################ portability #################

        return info_dict


    # TEST - compositonal - two lora + Connector(공통) - eval
    def test_sequencial_compositional_connector_eval(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") # 궁금한게, 200개에 대한 batch
        averager = RunningStatAverager("val")

        start_time = time.time()
        ## 저장할 내용
        val_data_store = []

        # visul-data
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        # textual-data
        base_logits_store_tex = []

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 1. Inference Output for test locality(visual & textual 데이터/출력 저장 & 출력)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                # 1.1) visual edit part
                val_data_store.append(batch) # batch 데이터 저장
                with torch.no_grad():
                    base_outputs = self.model(batch["visual_edit"]["loc"]) # T-Loc inference 저장 # self.model -> ft
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["visual_edit"]["loc_image"]) # I-Loc inference 저장
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach())

                # 1.2) textual edit part
                with torch.no_grad():
                    base_outputs = self.model(batch["textual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_tex.append(base_logits.clone().detach())

                pbar.update(1)
            else:
                break
        pbar.close()
                                                      
        ## 2. Model edit & Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            # 2.1) Model Edit (Update for a batch)
            # 2.1.1) Visual Edit(first)
            self.model.model.set_adapter("visual") # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["visual_edit"]["edit_inner"], mode = "visual" , peft = True)

            # 2.1.2) Textual Edit(second) 
            self.model.model.set_adapter("textual")  # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["textual_edit"]["edit_inner"], mode = "textual" , peft = True)

            # 2.2) Test with GAP
            if val_step >= gap_num: 
                # 기존 저장했던 batch, t-loc & i-loc-logit 불러옴. For Test
                stored_batch = val_data_store.pop(0) # vis + text
                stored_base_logits_vis = base_logits_store_vis.pop(0)
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0)
                stored_base_logits_tex = base_logits_store_tex.pop(0)

                # Test Sequential Edit(only inference & test) - vis / text 모두 다 평가해야 함.
                info_dict = self.test_sequencial_compositional_connector_eval_step(
                    stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_tex
                    )
                averager.add(info_dict)

            # logging?
            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log_CompositionalEdit( 
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Logging Results ## 
        if log:
            self._inline_seq_log_CompositionalEdit(val_step, averager.average(), start_time, steps) ## ★☆★ 수정 필요 ★☆★ ##
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        results_path = f"{self.config.adapter_path}/eval_wo_trained_weight/100/{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json"
        
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats
    

        ## TEST - compositonal - two
    
    def test_sequencial_compositional_connector_eval_50(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") # 궁금한게, 200개에 대한 batch
        averager = RunningStatAverager("val")

        start_time = time.time()
        ## 저장할 내용
        val_data_store = []

        # visul-data
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        # textual-data
        base_logits_store_tex = []

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 1. Inference Output for test locality(visual & textual 데이터/출력 저장 & 출력)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                # 1.1) visual edit part
                val_data_store.append(batch) # batch 데이터 저장
                with torch.no_grad():
                    base_outputs = self.model(batch["visual_edit"]["loc"]) # T-Loc inference 저장 # self.model -> ft
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["visual_edit"]["loc_image"]) # I-Loc inference 저장
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach())

                # 1.2) textual edit part
                with torch.no_grad():
                    base_outputs = self.model(batch["textual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_tex.append(base_logits.clone().detach())

                pbar.update(1)
            else:
                break
        pbar.close()
                                                      
        ## 2. Model edit & Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            # 2.1) Model Edit (Update for a batch)
            # 2.1.1) Visual Edit(first)
            self.model.model.set_adapter("visual") # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["visual_edit"]["edit_inner"], mode = "visual" , peft = True)

            # 2.1.2) Textual Edit(second) 
            self.model.model.set_adapter("textual")  # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["textual_edit"]["edit_inner"], mode = "textual" , peft = True)

            # 2.2) Test with GAP
            if val_step >= gap_num: 
                # 기존 저장했던 batch, t-loc & i-loc-logit 불러옴. For Test
                stored_batch = val_data_store.pop(0) # vis + text
                stored_base_logits_vis = base_logits_store_vis.pop(0)
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0)
                stored_base_logits_tex = base_logits_store_tex.pop(0)

                # Test Sequential Edit(only inference & test) - vis / text 모두 다 평가해야 함.
                info_dict = self.test_sequencial_compositional_connector_eval_step(
                    stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_tex
                    )
                averager.add(info_dict)

            # logging?
            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log_CompositionalEdit( 
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Logging Results ## 
        if log:
            self._inline_seq_log_CompositionalEdit(val_step, averager.average(), start_time, steps) ## ★☆★ 수정 필요 ★☆★ ##
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        results_path = f"{self.config.adapter_path}/eval_50/{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json"
        
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats
    
    def test_sequencial_compositional_connector_eval_25(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") # 궁금한게, 200개에 대한 batch
        averager = RunningStatAverager("val")

        start_time = time.time()
        ## 저장할 내용
        val_data_store = []

        # visul-data
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        # textual-data
        base_logits_store_tex = []

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 1. Inference Output for test locality(visual & textual 데이터/출력 저장 & 출력)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                # 1.1) visual edit part
                val_data_store.append(batch) # batch 데이터 저장
                with torch.no_grad():
                    base_outputs = self.model(batch["visual_edit"]["loc"]) # T-Loc inference 저장 # self.model -> ft
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["visual_edit"]["loc_image"]) # I-Loc inference 저장
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach())

                # 1.2) textual edit part
                with torch.no_grad():
                    base_outputs = self.model(batch["textual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_tex.append(base_logits.clone().detach())

                pbar.update(1)
            else:
                break
        pbar.close()
                                                      
        ## 2. Model edit & Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            # 2.1) Model Edit (Update for a batch)
            # 2.1.1) Visual Edit(first)
            self.model.model.set_adapter("visual") # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["visual_edit"]["edit_inner"], mode = "visual" , peft = True)

            # 2.1.2) Textual Edit(second) 
            self.model.model.set_adapter("textual")  # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["textual_edit"]["edit_inner"], mode = "textual" , peft = True)

            # 2.2) Test with GAP
            if val_step >= gap_num: 
                # 기존 저장했던 batch, t-loc & i-loc-logit 불러옴. For Test
                stored_batch = val_data_store.pop(0) # vis + text
                stored_base_logits_vis = base_logits_store_vis.pop(0)
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0)
                stored_base_logits_tex = base_logits_store_tex.pop(0)

                # Test Sequential Edit(only inference & test) - vis / text 모두 다 평가해야 함.
                info_dict = self.test_sequencial_compositional_connector_eval_step(
                    stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_tex
                    )
                averager.add(info_dict)

            # logging?
            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log_CompositionalEdit( 
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Logging Results ## 
        if log:
            self._inline_seq_log_CompositionalEdit(val_step, averager.average(), start_time, steps) ## ★☆★ 수정 필요 ★☆★ ##
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        results_path = f"{self.config.adapter_path}/eval_25/{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json"
        
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats
    

    def test_sequencial_compositional_connector_eval_step(self, batch, edited_model, base_logits_vis, base_image_logits_vis, base_logits_tex):
        info_dict = {}
        ## edited_model.model.lora_visual_activations
        ##############################################################################
        # ----------------------------Test: Visual Edit------------------------------#
        with torch.no_grad():
            # set lora: visual inference
            edited_model.model.set_adapter("visual")
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["visual_edit"]["edit_inner"])
            inner_batch_labels = batch["visual_edit"]["edit_inner"]["labels"]



            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels) # edit_loss_fn이 어딨지?
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(T-Gen)
            post_edit_outputs = edited_model(batch["visual_edit"]["edit_outer"])
            post_batch_labels = batch["visual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels) # edit_loss_fn -> vis, text 한번에 적용해도 되는가
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # image rephrase(I-Gen)
            post_image_edit_outputs = edited_model(batch["visual_edit"]["edit_outer_image"])
            post_image_batch_labels = batch["visual_edit"]["edit_outer_image"]["labels"]
            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
            else:
                post_image_edit_logits = post_image_edit_outputs

            if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)
            else:
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])
            del post_image_edit_outputs, post_image_edit_logits
            torch.cuda.empty_cache()

            # text loc(T-Loc)
            post_base_outputs = edited_model(batch["visual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_vis, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

            # image loc(I-Loc)
            post_image_base_outputs = edited_model(batch["visual_edit"]["loc_image"])
            if not isinstance(post_image_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
            else:
                post_image_base_logits = post_image_base_outputs
            post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
            base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits_vis, dim=-1), k=10, dim=-1).indices
            del post_image_base_outputs, post_image_base_logits
            torch.cuda.empty_cache()

        info_dict['vis/inner/acc'] = inner_edit_dict["acc"].item() # copy안해도 되는가? -> item은 int/float이 직접 반환됨. 따라서 ㅇㅇ
        info_dict['vis/edit/acc'] = post_edit_dict["acc"].item()
        info_dict['vis/image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["vis/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["vis/image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ##############################################################################
        # ----------------------------Test: Textual Edit------------------------------#
        with torch.no_grad():
            # set lora: visual inference
            edited_model.model.set_adapter("textual")
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["textual_edit"]["edit_inner"])
            inner_batch_labels = batch["textual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(Generality)
            post_edit_outputs = edited_model(batch["textual_edit"]["edit_outer"])
            post_batch_labels = batch["textual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # text loc(Locality)
            post_base_outputs = edited_model(batch["textual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_tex, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

        info_dict['text/inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['text/edit/acc'] = post_edit_dict["acc"].item()
        info_dict["text/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ################ portability #################
        # set lora: visual&textual inference
        edited_model.model.set_adapter(["textual","visual","connector"])

        if batch['port'] is not None:
            assert len(batch['port']) == 1, "batch['port'] exist and have only one element"
            port = batch['port'][0]
            with torch.no_grad():
                port_outputs = edited_model(port)
                port_labels = port["labels"]
                if not isinstance(port_outputs, torch.Tensor):
                    port_logits = port_outputs.logits
                else:
                    port_logits = port_outputs
                if port_logits.shape[1] > port_labels.shape[1]:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
                else:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
                port_acc = port_dict["acc"].item()
                del port_outputs, port_logits
                torch.cuda.empty_cache()

            info_dict['port/acc'] = port_acc
            info_dict['port/ratio'] =  port_acc / (info_dict['vis/inner/acc'] + info_dict['text/inner/acc'] + 1e-8) * 2
            ################ portability #################

        return info_dict

    # TEST - compositonal - two lora + Connec tor(공통) - eval
    def test_sequencial_compositional_connector_eval_vis(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") # 궁금한게, 200개에 대한 batch
        averager = RunningStatAverager("val")

        start_time = time.time()
        ## 저장할 내용
        val_data_store = []

        # visul-data
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        # textual-data
        base_logits_store_tex = []

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 1. Inference Output for test locality(visual & textual 데이터/출력 저장 & 출력)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                # 1.1) visual edit part
                val_data_store.append(batch) # batch 데이터 저장
                with torch.no_grad():
                    base_outputs = self.model(batch["visual_edit"]["loc"]) # T-Loc inference 저장 # self.model -> ft
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["visual_edit"]["loc_image"]) # I-Loc inference 저장
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach())

                # 1.2) textual edit part
                with torch.no_grad():
                    base_outputs = self.model(batch["textual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_tex.append(base_logits.clone().detach())

                pbar.update(1)
            else:
                break
        pbar.close()
                                                      
        ## 2. Model edit & Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            # 2.1) Model Edit (Update for a batch)
            # 2.1.1) Visual Edit(first)
            self.model.model.set_adapter("visual") # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["visual_edit"]["edit_inner"], mode = "visual" , peft = True)

            # 2.1.2) Textual Edit(second) 
            self.model.model.set_adapter("textual")  # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["textual_edit"]["edit_inner"], mode = "textual" , peft = True)

            # 2.2) Test with GAP
            if val_step >= gap_num: 
                # 기존 저장했던 batch, t-loc & i-loc-logit 불러옴. For Test
                stored_batch = val_data_store.pop(0) # vis + text
                stored_base_logits_vis = base_logits_store_vis.pop(0)
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0)
                stored_base_logits_tex = base_logits_store_tex.pop(0)

                # Test Sequential Edit(only inference & test) - vis / text 모두 다 평가해야 함.
                info_dict = self.test_sequencial_compositional_connector_eval_vis_step(
                    stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_tex
                    )
                averager.add(info_dict)

            # logging?
            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log_CompositionalEdit( 
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Logging Results ## 
        if log:
            self._inline_seq_log_CompositionalEdit(val_step, averager.average(), start_time, steps) ## ★☆★ 수정 필요 ★☆★ ##
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        results_path = f"{self.config.adapter_path}/eval/{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json"
        
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats
    

        ## TEST - compositonal - two
    def test_sequencial_compositional_connector_eval_vis_step(self, batch, edited_model, base_logits_vis, base_image_logits_vis, base_logits_tex):
        info_dict = {}
        ## edited_model.model.lora_visual_activations
        ##############################################################################
        # ----------------------------Test: Visual Edit------------------------------#
        with torch.no_grad():
            # set lora: visual inference
            edited_model.model.set_adapter("visual")
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["visual_edit"]["edit_inner"])
            inner_batch_labels = batch["visual_edit"]["edit_inner"]["labels"]

            # vis lora
            # edited_model.model.set_adapter("textual")
            # inner_edit_outputs = edited_model(batch["visual_edit"]["edit_inner"])
            # self.visualization_heatmap_texlora(edited_model)
            # edited_model.model.set_adapter("visual")

            # comp lora
            edited_model.model.set_adapter(["visual","textual"])

            # 모델 활성화 dictionary 초기화 (inference 전에 이전 값 제거)
            if hasattr(edited_model.model, 'lora_visual_activations'):
                edited_model.model.lora_visual_activations.clear()
            else:
                edited_model.model.lora_visual_activations = {}

            inner_edit_outputs = edited_model(batch["visual_edit"]["edit_inner"])
            self.visualization_heatmap_lora_compare_visdata(edited_model)
            #self.visualization_heatmap_lora_compare_visdata_renew(edited_model)
            edited_model.model.set_adapter("visual")


            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels) # edit_loss_fn이 어딨지?
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(T-Gen)
            post_edit_outputs = edited_model(batch["visual_edit"]["edit_outer"])
            post_batch_labels = batch["visual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels) # edit_loss_fn -> vis, text 한번에 적용해도 되는가
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # image rephrase(I-Gen)
            post_image_edit_outputs = edited_model(batch["visual_edit"]["edit_outer_image"])
            post_image_batch_labels = batch["visual_edit"]["edit_outer_image"]["labels"]
            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
            else:
                post_image_edit_logits = post_image_edit_outputs

            if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)
            else:
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])
            del post_image_edit_outputs, post_image_edit_logits
            torch.cuda.empty_cache()

            # text loc(T-Loc)
            post_base_outputs = edited_model(batch["visual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_vis, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

            # image loc(I-Loc)
            post_image_base_outputs = edited_model(batch["visual_edit"]["loc_image"])
            if not isinstance(post_image_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
            else:
                post_image_base_logits = post_image_base_outputs
            post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
            base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits_vis, dim=-1), k=10, dim=-1).indices
            del post_image_base_outputs, post_image_base_logits
            torch.cuda.empty_cache()

        info_dict['vis/inner/acc'] = inner_edit_dict["acc"].item() # copy안해도 되는가? -> item은 int/float이 직접 반환됨. 따라서 ㅇㅇ
        info_dict['vis/edit/acc'] = post_edit_dict["acc"].item()
        info_dict['vis/image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["vis/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["vis/image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ##############################################################################
        # ----------------------------Test: Textual Edit------------------------------#
        with torch.no_grad():
            # set lora: visual inference
            edited_model.model.set_adapter("textual")
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["textual_edit"]["edit_inner"])
            inner_batch_labels = batch["textual_edit"]["edit_inner"]["labels"]

            # comp lora
            edited_model.model.set_adapter(["visual","textual"])

            # 모델 활성화 dictionary 초기화 (inference 전에 이전 값 제거)
            if hasattr(edited_model.model, 'lora_visual_activations'):
                edited_model.model.lora_visual_activations.clear()
            else:
                edited_model.model.lora_visual_activations = {}

            inner_edit_outputs = edited_model(batch["textual_edit"]["edit_inner"])
            self.visualization_heatmap_lora_compare_texdata(edited_model)
            edited_model.model.set_adapter("textual")


            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(Generality)
            post_edit_outputs = edited_model(batch["textual_edit"]["edit_outer"])
            post_batch_labels = batch["textual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # text loc(Locality)
            post_base_outputs = edited_model(batch["textual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_tex, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

        info_dict['text/inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['text/edit/acc'] = post_edit_dict["acc"].item()
        info_dict["text/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ################ portability #################
        # set lora: visual&textual inference
        edited_model.model.set_adapter(["textual","visual","connector"])

        if batch['port'] is not None:
            assert len(batch['port']) == 1, "batch['port'] exist and have only one element"
            port = batch['port'][0]
            with torch.no_grad():
                port_outputs = edited_model(port)
                port_labels = port["labels"]
                if not isinstance(port_outputs, torch.Tensor):
                    port_logits = port_outputs.logits
                else:
                    port_logits = port_outputs
                if port_logits.shape[1] > port_labels.shape[1]:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
                else:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
                port_acc = port_dict["acc"].item()
                del port_outputs, port_logits
                torch.cuda.empty_cache()

            info_dict['port/acc'] = port_acc
            info_dict['port/ratio'] =  port_acc / (info_dict['vis/inner/acc'] + info_dict['text/inner/acc'] + 1e-8) * 2
            ################ portability #################

        return info_dict

    def visualization_heatmap_vislora(self, model):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        num_layers = 32
        layer_keys = [f"layer{i}.up_proj.lora_B.visual" for i in range(num_layers)]

        token_activation_summary = []

        lora_visual_activations = model.model.lora_visual_activations

        for key in layer_keys:
            if key in lora_visual_activations:
                activation = lora_visual_activations[key][0]  # shape: [num_tokens, hidden_dim]
                token_summary = np.mean(activation, axis=1)

                token_activation_summary.append(token_summary)
            else:
                token_activation_summary.append(np.zeros(593))

        data = np.stack(token_activation_summary, axis=0)

        plt.figure(figsize=(15, 10))
        sns.heatmap(data, cmap='viridis')  # 0~1 범위로 조정
        plt.title("Activation Heatmap: Visual Adapter Layers vs Input Tokens")
        plt.xlabel("Input Token Index")
        plt.ylabel("Layer Index")
        plt.savefig("vis_layers_tokens_activation_heatmap_mean.png", dpi=300, bbox_inches='tight')
        plt.savefig("vis_layers_tokens_activation_heatmap_mean.pdf", dpi=300, bbox_inches='tight')
    
    def visualization_heatmap_texlora(self, model):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        num_layers = 32
        layer_keys = [f"layer{i}.up_proj.lora_B.textual" for i in range(num_layers)]

        token_activation_summary = []

        lora_visual_activations = model.model.lora_visual_activations

        for key in layer_keys:
            if key in lora_visual_activations:
                activation = lora_visual_activations[key][0]  # shape: [num_tokens, hidden_dim]
                token_summary = np.mean(activation, axis=1)

                token_activation_summary.append(token_summary)
            else:
                token_activation_summary.append(np.zeros(593))

        data = np.stack(token_activation_summary, axis=0)

        plt.figure(figsize=(15, 10))
        sns.heatmap(data, cmap='viridis')  # 0~1 범위로 조정
        plt.title("Activation Heatmap: Textual Adapter Layers vs Input Tokens")
        plt.xlabel("Input Token Index")
        plt.ylabel("Layer Index")
        plt.savefig("tex_layers_tokens_activation_heatmap_mean.png", dpi=300, bbox_inches='tight')
        plt.savefig("tex_layers_tokens_activation_heatmap_mean.pdf", dpi=300, bbox_inches='tight')

    def visualization_heatmap_lora_compare_visdata(self, model):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        num_layers = 32
        num_tokens = 593  # 예시
        
        # Visual LoRA 키 / Textual LoRA 키
        visual_layer_keys = [f"layer{i}.up_proj.lora_A.visual" for i in range(num_layers)] # up_proj.lora_A
        textual_layer_keys = [f"layer{i}.up_proj.lora_A.textual" for i in range(num_layers)]
        
        lora_visual_activations = model.model.lora_visual_activations
        
        # Visual / Textual 각각 활성화 요약을 담을 리스트
        visual_activation_summary = []
        textual_activation_summary = []
        
        # Visual LoRA 활성화 추출
        for key in visual_layer_keys:
            if key in lora_visual_activations:
                activation = lora_visual_activations[key][0]  # shape: [num_tokens, hidden_dim]
                token_summary = np.mean(activation, axis=1)   # 토큰별 평균

            else:
                token_summary = np.zeros(num_tokens)
            visual_activation_summary.append(token_summary)
        
        # Textual LoRA 활성화 추출
        for key in textual_layer_keys:
            if key in lora_visual_activations:
                activation = lora_visual_activations[key][0]  # shape: [num_tokens, hidden_dim]
                token_summary = np.mean(activation, axis=1)


            else:
                token_summary = np.zeros(num_tokens)
            textual_activation_summary.append(token_summary)
        
        # (num_layers, num_tokens) shape으로 변환
        visual_data = np.stack(visual_activation_summary, axis=0)
        textual_data = np.stack(textual_activation_summary, axis=0)

        global_min = min(visual_data.min(), textual_data.min())
        global_max = max(visual_data.max(), textual_data.max())

        # 시각화
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))
        
        # 시각적 비교를 위해 vmin, vmax를 동일하게 설정 (예: -0.5 ~ 0.5)
        sns.heatmap(visual_data, cmap='viridis', ax=ax1, vmin=global_min, vmax=global_max)
        ax1.set_title("Activation Heatmap: Visual LoRA (Layers vs Tokens)")
        ax1.set_xlabel("Token Index")
        ax1.set_ylabel("Layer Index")
        # y축 범위 통일 (0 ~ num_layers)
        ax1.set_ylim(0, num_layers)
        # y축 눈금도 0~31로 맞추고 싶다면 다음과 같이 설정 가능
        # ax1.set_yticks(range(num_layers))
        # ax1.set_yticklabels(range(num_layers))
        
        sns.heatmap(textual_data, cmap='viridis', ax=ax2, vmin=global_min, vmax=global_max)
        ax2.set_title("Activation Heatmap: Textual LoRA (Layers vs Tokens)")
        ax2.set_xlabel("Token Index")
        ax2.set_ylabel("Layer Index")
        ax2.set_ylim(0, num_layers)
        # ax2.set_yticks(range(num_layers))
        # ax2.set_yticklabels(range(num_layers))
        
        plt.tight_layout()
        plt.savefig("visdata_lora_visual_textual_heatmap_compare.png", dpi=300, bbox_inches='tight')
        plt.savefig("visdata_lora_visual_textual_heatmap_compare.pdf", dpi=300, bbox_inches='tight')
        plt.show()

    def visualization_heatmap_lora_compare_texdata(self, model):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        num_layers = 32
        num_tokens = 593  # 예시
        
        # Visual LoRA 키 / Textual LoRA 키
        visual_layer_keys = [f"layer{i}.up_proj.lora_A.visual" for i in range(num_layers)]
        textual_layer_keys = [f"layer{i}.up_proj.lora_A.textual" for i in range(num_layers)]
        
        lora_visual_activations = model.model.lora_visual_activations
        
        # Visual / Textual 각각 활성화 요약을 담을 리스트
        visual_activation_summary = []
        textual_activation_summary = []
        
        # Visual LoRA 활성화 추출
        for key in visual_layer_keys:
            if key in lora_visual_activations:
                activation = lora_visual_activations[key][0]  # shape: [num_tokens, hidden_dim]
                token_summary = np.mean(activation, axis=1)   # 토큰별 평균

                # min_val = np.min(token_summary)
                # max_val = np.max(token_summary)

                # if max_val > min_val:  # NaN 방지
                #     token_summary = (token_summary - min_val) / (max_val - min_val)
                
            else:
                token_summary = np.zeros(num_tokens)
            visual_activation_summary.append(token_summary)
        
        # Textual LoRA 활성화 추출
        for key in textual_layer_keys:
            if key in lora_visual_activations:
                activation = lora_visual_activations[key][0]  # shape: [num_tokens, hidden_dim]
                token_summary = np.mean(activation, axis=1)

                # min_val = np.min(token_summary)
                # max_val = np.max(token_summary)

                # if max_val > min_val:  # NaN 방지
                #     token_summary = (token_summary - min_val) / (max_val - min_val)

            else:
                token_summary = np.zeros(num_tokens)
            textual_activation_summary.append(token_summary)
        
        # (num_layers, num_tokens) shape으로 변환
        visual_data = np.stack(visual_activation_summary, axis=0)
        textual_data = np.stack(textual_activation_summary, axis=0)

        global_min = min(visual_data.min(), textual_data.min())
        global_max = max(visual_data.max(), textual_data.max())


        # 시각화
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))
        
        # 시각적 비교를 위해 vmin, vmax를 동일하게 설정 (예: -0.5 ~ 0.5)
        sns.heatmap(visual_data, cmap='viridis', ax=ax1, vmin=global_min, vmax=global_max)
        ax1.set_title("Activation Heatmap: Visual LoRA (Layers vs Tokens)")
        ax1.set_xlabel("Token Index")
        ax1.set_ylabel("Layer Index")
        # y축 범위 통일 (0 ~ num_layers)
        ax1.set_ylim(0, num_layers)
        # y축 눈금도 0~31로 맞추고 싶다면 다음과 같이 설정 가능
        # ax1.set_yticks(range(num_layers))
        # ax1.set_yticklabels(range(num_layers))
        
        sns.heatmap(textual_data, cmap='viridis', ax=ax2, vmin=global_min, vmax=global_max)
        ax2.set_title("Activation Heatmap: Textual LoRA (Layers vs Tokens)")
        ax2.set_xlabel("Token Index")
        ax2.set_ylabel("Layer Index")
        ax2.set_ylim(0, num_layers)
        # ax2.set_yticks(range(num_layers))
        # ax2.set_yticklabels(range(num_layers))
        
        plt.tight_layout()
        plt.savefig("texdata_lora_visual_textual_heatmap_compare.png", dpi=300, bbox_inches='tight')
        plt.savefig("texdata_lora_visual_textual_heatmap_compare.pdf", dpi=300, bbox_inches='tight')
        plt.show()

    def visualization_heatmap_lora_compare_visdata_renew(self, model):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import datetime
        import logging

        # 로깅 설정 (원하는 로깅 설정을 추가할 수 있음)
        logging.basicConfig(level=logging.INFO)


        # 하드코딩된 파라미터 (필요시 동적으로 얻도록 수정 가능)
        num_layers = 32      # 예시: 총 레이어 수
        num_tokens = 593     # 예시: 토큰 수 (입력 데이터에 맞게 변경)

        # Visual / Textual LoRA의 각 레이어에 해당하는 키 생성
        visual_layer_keys = [f"layer{i}.up_proj.lora_A.visual" for i in range(num_layers)]
        textual_layer_keys = [f"layer{i}.up_proj.lora_A.textual" for i in range(num_layers)]

        # hook에서 저장된 활성화 데이터 dictionary
        lora_visual_activations = model.model.lora_visual_activations

        # 각 도메인의 활성화 요약 (토큰별 평균값) 저장 리스트
        visual_activation_summary = []
        textual_activation_summary = []

        # Visual LoRA 활성화 추출
        for key in visual_layer_keys:
            if key in lora_visual_activations:
                try:
                    # activation shape: [num_tokens, hidden_dim]
                    activation = lora_visual_activations[key][0]
                    token_summary = np.mean(activation, axis=1)  # 각 토큰에 대한 평균값 계산
                except Exception as e:
                    logging.warning(f"키 {key}의 활성화 처리 중 에러 발생: {e}")
                    token_summary = np.zeros(num_tokens)
            else:
                logging.warning(f"키 {key}가 활성화 데이터에 없습니다. 0으로 채웁니다.")
                token_summary = np.zeros(num_tokens)
            visual_activation_summary.append(token_summary)

        # Textual LoRA 활성화 추출
        for key in textual_layer_keys:
            if key in lora_visual_activations:
                try:
                    activation = lora_visual_activations[key][0]
                    token_summary = np.mean(activation, axis=1)
                except Exception as e:
                    logging.warning(f"키 {key}의 활성화 처리 중 에러 발생: {e}")
                    token_summary = np.zeros(num_tokens)
            else:
                logging.warning(f"키 {key}가 활성화 데이터에 없습니다. 0으로 채웁니다.")
                token_summary = np.zeros(num_tokens)
            textual_activation_summary.append(token_summary)

        # (num_layers, num_tokens) shape의 2D 배열로 변환
        visual_data = np.stack(visual_activation_summary, axis=0)
        textual_data = np.stack(textual_activation_summary, axis=0)

        # 두 도메인에 대한 global 최소/최대 값 산출 (색상 스케일 통일)
        global_min = min(visual_data.min(), textual_data.min())
        global_max = max(visual_data.max(), textual_data.max())

        # 파일명 중복 방지를 위한 타임스탬프 생성
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 시각화: 두 heatmap을 나란히 비교
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))

        sns.heatmap(visual_data, cmap='viridis', ax=ax1, vmin=global_min, vmax=global_max)
        ax1.set_title("Activation Heatmap: Visual LoRA (Layers vs Tokens)")
        ax1.set_xlabel("Token Index")
        ax1.set_ylabel("Layer Index")
        ax1.set_ylim(0, num_layers)
        ax1.set_yticks(range(num_layers))
        ax1.set_yticklabels(range(num_layers))

        sns.heatmap(textual_data, cmap='viridis', ax=ax2, vmin=global_min, vmax=global_max)
        ax2.set_title("Activation Heatmap: Textual LoRA (Layers vs Tokens)")
        ax2.set_xlabel("Token Index")
        ax2.set_ylabel("Layer Index")
        ax2.set_ylim(0, num_layers)
        ax2.set_yticks(range(num_layers))
        ax2.set_yticklabels(range(num_layers))

        plt.tight_layout()

        # 결과 파일 저장 (PNG 및 PDF)
        png_filename = f"visdata_lora_visual_textual_heatmap_compare_{timestamp}.png"
        pdf_filename = f"visdata_lora_visual_textual_heatmap_compare_{timestamp}.pdf"
        plt.savefig(png_filename, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_filename, dpi=300, bbox_inches='tight')
        plt.show()

        ## min - max 
        # def visualization_heatmap_vislora(self, model):
        #     import matplotlib.pyplot as plt
        #     import seaborn as sns
        #     import numpy as np
            
        #     num_layers = 32
        #     layer_keys = [f"layer{i}.up_proj.lora_B.visual" for i in range(num_layers)]

        #     token_activation_summary = []

        #     lora_visual_activations = model.model.lora_visual_activations

        #     for key in layer_keys:
        #         if key in lora_visual_activations:
        #             activation = lora_visual_activations[key][0]  # shape: [num_tokens, hidden_dim]
        #             token_summary = np.mean(activation, axis=1)

        #             # Layer-wise Min-Max Scaling
        #             min_val = np.min(token_summary)
        #             max_val = np.max(token_summary)

        #             if max_val > min_val:  # NaN 방지
        #                 token_summary = (token_summary - min_val) / (max_val - min_val)

        #             token_activation_summary.append(token_summary)
        #         else:
        #             token_activation_summary.append(np.zeros(593))

        #     data = np.stack(token_activation_summary, axis=0)

        #     plt.figure(figsize=(15, 10))
        #     sns.heatmap(data, cmap='viridis', vmin=0, vmax=1)  # 0~1 범위로 조정
        #     plt.title("Activation Heatmap: Visual Adapter Layers vs Input Tokens (Min-Max Normalized)")
        #     plt.xlabel("Input Token Index")
        #     plt.ylabel("Layer Index")
        #     plt.savefig("vis_layers_tokens_activation_heatmap_minmax.png", dpi=300, bbox_inches='tight')
        #     plt.savefig("vis_layers_tokens_activation_heatmap_minmax.pdf", dpi=300, bbox_inches='tight')
        # def visualization_heatmap_texlora(self, model):
        #     import matplotlib.pyplot as plt
        #     import seaborn as sns
        #     import numpy as np
            
        #     num_layers = 32
        #     layer_keys = [f"layer{i}.up_proj.lora_B.textual" for i in range(num_layers)]

        #     token_activation_summary = []

        #     lora_visual_activations = model.model.lora_visual_activations

        #     for key in layer_keys:
        #         if key in lora_visual_activations:
        #             activation = lora_visual_activations[key][0]  # shape: [num_tokens, hidden_dim]
        #             token_summary = np.mean(activation, axis=1)

        #             # Layer-wise Min-Max Scaling
        #             min_val = np.min(token_summary)
        #             max_val = np.max(token_summary)

        #             if max_val > min_val:  # NaN 방지
        #                 token_summary = (token_summary - min_val) / (max_val - min_val)

        #             token_activation_summary.append(token_summary)
        #         else:
        #             token_activation_summary.append(np.zeros(593))

        #     data = np.stack(token_activation_summary, axis=0)

        #     plt.figure(figsize=(15, 10))
        #     sns.heatmap(data, cmap='viridis', vmin=0, vmax=1)  # 0~1 범위로 조정
        #     plt.title("Activation Heatmap: Textual Adapter Layers vs Input Tokens (Min-Max Normalized)")
        #     plt.xlabel("Input Token Index")
        #     plt.ylabel("Layer Index")
        #     plt.savefig("tex_layers_tokens_activation_heatmap_minmax.png", dpi=300, bbox_inches='tight')
        #     plt.savefig("tex_layers_tokens_activation_heatmap_minmax.pdf", dpi=300, bbox_inches='tight')




        #     ###############################################################
        
        
    ### --- TEST - compositonal - two lora + Connector(att) --- ###
    def test_sequencial_compositional_connector_attention(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") # 궁금한게, 200개에 대한 batch
        averager = RunningStatAverager("val")

        start_time = time.time()
        ## 저장할 내용
        val_data_store = []

        # visul-data
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        # textual-data
        base_logits_store_tex = []

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 1. Inference Output for test locality(visual & textual 데이터/출력 저장 & 출력)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                # 1.1) visual edit part
                val_data_store.append(batch) # batch 데이터 저장
                with torch.no_grad():
                    base_outputs = self.model(batch["visual_edit"]["loc"]) # T-Loc inference 저장 # self.model -> ft
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["visual_edit"]["loc_image"]) # I-Loc inference 저장
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach())

                # 1.2) textual edit part
                with torch.no_grad():
                    base_outputs = self.model(batch["textual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_tex.append(base_logits.clone().detach())

                pbar.update(1)
            else:
                break
        pbar.close()

        ## 2. Model edit & Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            # 2.1) Model Edit (Update for a batch)
            # 2.1.1) Visual Edit(first)
            self.model.model.set_adapter("visual") # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["visual_edit"]["edit_inner"], mode = "visual" , peft = True)

            # 2.1.2) Textual Edit(second) 
            self.model.model.set_adapter("textual")  # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["textual_edit"]["edit_inner"], mode = "textual" , peft = True)

            # 2.1.3) Compositional Edit(second) ★ mlp 학습 o
            if val_step > 5:
                edited_model.model.set_adapter(["textual","visual","connector"])
                edited_model, _ = edited_model.edit(batch["port"][0], connector_mode=True) # cond? 이거 안되나


            # 2.2) Test with GAP
            if val_step >= gap_num: 
                # 기존 저장했던 batch, t-loc & i-loc-logit 불러옴. For Test
                stored_batch = val_data_store.pop(0) # vis + text
                stored_base_logits_vis = base_logits_store_vis.pop(0)
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0)
                stored_base_logits_tex = base_logits_store_tex.pop(0)

                # Test Sequential Edit(only inference & test) - vis / text 모두 다 평가해야 함.
                info_dict = self.test_sequencial_compositional_connector_attention_step(
                    stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_tex
                    )
                averager.add(info_dict)

            # logging?
            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log_CompositionalEdit( ## ★☆★ 수정 필요 ★☆★ ##
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Logging Results ## 
        if log:
            self._inline_seq_log_CompositionalEdit(val_step, averager.average(), start_time, steps) ## ★☆★ 수정 필요 ★☆★ ##
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        results_path = f"results/results_sequencial/composition/two_lora_connect_attention/{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json"
        
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        if gap_num == 0:
            try: # lora weight 저장
                from peft import LoraConfig, TaskType, get_peft_model, PeftConfig, PeftModel
                connector_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=8,
                        lora_alpha=16,
                        lora_dropout=0.05,
                        target_modules=["q_proj", "k_proj"]
                    )
                
                peft_model = get_peft_model(self.model.model.base_model.model, connector_config)
                peft_model.delete_adapter("default")
                peft_model = peft_model.cpu()
                peft_model.save_pretrained("results/results_sequencial/composition/two_lora_connect_attention")
                # 저장 후 메모리 해제
                del peft_model

                torch.cuda.empty_cache()
                print("LoRA + (gap0, train_composition.json) 모델 저장 완료 -> \"results/results_sequencial/composition/two_lora_connect_attention\" ")
            except:
                    print("LoRA, MLP 모델 저장 실패")

            with open(results_path, "w") as f:
                json.dump(
                    {"results": stats}, f
                )
                LOG.info("Wrote results to:")
                LOG.info(results_path)

            return stats
        

            ## TEST - compositonal - two
    def test_sequencial_compositional_connector_attention_step(self, batch, edited_model, base_logits_vis, base_image_logits_vis, base_logits_tex):
            info_dict = {}

            ##############################################################################
            # ----------------------------Test: Visual Edit------------------------------#
            with torch.no_grad():
                # set lora: visual inference
                edited_model.model.set_adapter("visual")
                # inner(Reliability)
                inner_edit_outputs = edited_model(batch["visual_edit"]["edit_inner"])
                inner_batch_labels = batch["visual_edit"]["edit_inner"]["labels"]
                if not isinstance(inner_edit_outputs, torch.Tensor):
                    inner_edit_logits = inner_edit_outputs.logits
                else:
                    inner_edit_logits = inner_edit_outputs

                if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                    inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels) # edit_loss_fn이 어딨지?
                else:
                    inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
                del inner_edit_outputs, inner_edit_logits
                torch.cuda.empty_cache()

                # text rephrase(T-Gen)
                post_edit_outputs = edited_model(batch["visual_edit"]["edit_outer"])
                post_batch_labels = batch["visual_edit"]["edit_outer"]["labels"]
                if not isinstance(post_edit_outputs, torch.Tensor):
                    post_edit_logits = post_edit_outputs.logits
                else:
                    post_edit_logits = post_edit_outputs
                
                if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                    post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels) # edit_loss_fn -> vis, text 한번에 적용해도 되는가
                else:
                    post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
                del post_edit_outputs, post_edit_logits
                torch.cuda.empty_cache()

                # image rephrase(I-Gen)
                post_image_edit_outputs = edited_model(batch["visual_edit"]["edit_outer_image"])
                post_image_batch_labels = batch["visual_edit"]["edit_outer_image"]["labels"]
                if not isinstance(post_image_edit_outputs, torch.Tensor):
                    post_image_edit_logits = post_image_edit_outputs.logits
                else:
                    post_image_edit_logits = post_image_edit_outputs

                if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                    image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)
                else:
                    image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])
                del post_image_edit_outputs, post_image_edit_logits
                torch.cuda.empty_cache()

                # text loc(T-Loc)
                post_base_outputs = edited_model(batch["visual_edit"]["loc"])
                if not isinstance(post_base_outputs, torch.Tensor):
                    post_base_logits = post_base_outputs.logits
                else:
                    post_base_logits = post_base_outputs
                post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
                base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_vis, dim=-1), k=1, dim=-1).indices
                del post_base_outputs, post_base_logits
                torch.cuda.empty_cache()

                # image loc(I-Loc)
                post_image_base_outputs = edited_model(batch["visual_edit"]["loc_image"])
                if not isinstance(post_image_base_outputs, torch.Tensor):
                    post_image_base_logits = post_image_base_outputs.logits
                else:
                    post_image_base_logits = post_image_base_outputs
                post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
                base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits_vis, dim=-1), k=10, dim=-1).indices
                del post_image_base_outputs, post_image_base_logits
                torch.cuda.empty_cache()

            info_dict['vis/inner/acc'] = inner_edit_dict["acc"].item() # copy안해도 되는가? -> item은 int/float이 직접 반환됨. 따라서 ㅇㅇ
            info_dict['vis/edit/acc'] = post_edit_dict["acc"].item()
            info_dict['vis/image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
            info_dict["vis/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
            info_dict["vis/image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
            ##############################################################################

            ##############################################################################
            # ----------------------------Test: Textual Edit------------------------------#
            with torch.no_grad():
                # set lora: visual inference
                edited_model.model.set_adapter("textual")
                # inner(Reliability)
                inner_edit_outputs = edited_model(batch["textual_edit"]["edit_inner"])
                inner_batch_labels = batch["textual_edit"]["edit_inner"]["labels"]
                if not isinstance(inner_edit_outputs, torch.Tensor):
                    inner_edit_logits = inner_edit_outputs.logits
                else:
                    inner_edit_logits = inner_edit_outputs

                if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                    inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
                else:
                    inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
                del inner_edit_outputs, inner_edit_logits
                torch.cuda.empty_cache()

                # text rephrase(Generality)
                post_edit_outputs = edited_model(batch["textual_edit"]["edit_outer"])
                post_batch_labels = batch["textual_edit"]["edit_outer"]["labels"]
                if not isinstance(post_edit_outputs, torch.Tensor):
                    post_edit_logits = post_edit_outputs.logits
                else:
                    post_edit_logits = post_edit_outputs
                
                if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                    post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
                else:
                    post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
                del post_edit_outputs, post_edit_logits
                torch.cuda.empty_cache()

                # text loc(Locality)
                post_base_outputs = edited_model(batch["textual_edit"]["loc"])
                if not isinstance(post_base_outputs, torch.Tensor):
                    post_base_logits = post_base_outputs.logits
                else:
                    post_base_logits = post_base_outputs
                post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
                base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_tex, dim=-1), k=1, dim=-1).indices
                del post_base_outputs, post_base_logits
                torch.cuda.empty_cache()

            info_dict['text/inner/acc'] = inner_edit_dict["acc"].item()
            info_dict['text/edit/acc'] = post_edit_dict["acc"].item()
            info_dict["text/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
            ##############################################################################

            ################ portability #################
            # set lora: visual&textual inference
            edited_model.model.set_adapter(["textual","visual","connector"])

            if batch['port'] is not None:
                assert len(batch['port']) == 1, "batch['port'] exist and have only one element"
                port = batch['port'][0]
                with torch.no_grad():
                    port_outputs = edited_model(port)
                    port_labels = port["labels"]
                    if not isinstance(port_outputs, torch.Tensor):
                        port_logits = port_outputs.logits
                    else:
                        port_logits = port_outputs
                    if port_logits.shape[1] > port_labels.shape[1]:
                        port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
                    else:
                        port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
                    port_acc = port_dict["acc"].item()
                    del port_outputs, port_logits
                    torch.cuda.empty_cache()

                info_dict['port/acc'] = port_acc
                ################ portability #################

            return info_dict

    #     ## TEST - compositonal - two
    def test_sequencial_compositional_connector_attention_rag_70(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") # 궁금한게, 200개에 대한 batch
        averager = RunningStatAverager("val")

        start_time = time.time()
        ## 저장할 내용
        val_data_store = []

        # visul-data
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        # textual-data
        base_logits_store_tex = []

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 1. Inference Output for test locality(visual & textual 데이터/출력 저장 & 출력)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                # 1.1) visual edit part
                val_data_store.append(batch) # batch 데이터 저장
                with torch.no_grad():
                    base_outputs = self.model(batch["visual_edit"]["loc"]) # T-Loc inference 저장 # self.model -> ft
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["visual_edit"]["loc_image"]) # I-Loc inference 저장
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach())

                # 1.2) textual edit part
                with torch.no_grad():
                    base_outputs = self.model(batch["textual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_tex.append(base_logits.clone().detach())

                pbar.update(1)
            else:
                break
        pbar.close()

        ## 2. Model edit & Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            # 2.1) Model Edit (Update for a batch)
            # 2.1.1) Visual Edit(first)
            self.model.model.set_adapter("visual") # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["visual_edit"]["edit_inner"], mode = "visual" , peft = True)

            # 2.1.2) Textual Edit(second) 
            self.model.model.set_adapter("textual")  # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["textual_edit"]["edit_inner"], mode = "textual" , peft = True)

            # 2.1.3) Compositional Edit(second) ★ mlp 학습 o
            if val_step > 5:
                edited_model.model.set_adapter(["textual","visual","connector"])
                edited_model, _ = edited_model.edit(batch["port"][0], connector_mode=True) # cond? 이거 안되나


            # 2.2) Test with GAP
            if val_step >= gap_num: 
                # 기존 저장했던 batch, t-loc & i-loc-logit 불러옴. For Test
                stored_batch = val_data_store.pop(0) # vis + text
                stored_base_logits_vis = base_logits_store_vis.pop(0)
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0)
                stored_base_logits_tex = base_logits_store_tex.pop(0)

                # Test Sequential Edit(only inference & test) - vis / text 모두 다 평가해야 함.
                info_dict = self.test_sequencial_compositional_connector_attention_rag_step(
                    stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_tex
                    )
                averager.add(info_dict)

            # logging?
            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                    self._inline_seq_log_CompositionalEdit( ## ★☆★ 수정 필요 ★☆★ ##
                        val_step, averager.average(), start_time, steps
                    )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Logging Results ## 
        if log:
            self._inline_seq_log_CompositionalEdit(val_step, averager.average(), start_time, steps) ## ★☆★ 수정 필요 ★☆★ ##
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps


        if 'llava' in self.config.model_name.lower():
            result_dir = f"results/results_sequencial/composition/two_lora_connect_attention_rag_70"
        elif 'blip2' in self.config.model_name.lower() :
            result_dir = f"results/results_sequencial/blip2/composition/two_lora_connect_attention_rag_70"
        elif 'minigpt4' in self.config.model_name.lower() :
            result_dir = f"results/results_sequencial/minigpt4/composition/two_lora_connect_attention_rag_70"
        
        results_path = os.path.join(result_dir, f"{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json")
        
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        if gap_num == 0:
            try: # lora weight 저장
                from peft import LoraConfig, TaskType, get_peft_model, PeftConfig, PeftModel
                connector_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=8,
                        lora_alpha=16,
                        lora_dropout=0.05,
                        target_modules=["q_proj", "k_proj"]
                    )
                
                
                # blip이면
                if 'llava' in self.config.model_name.lower() :
                    peft_model = get_peft_model(self.model.model.base_model.model, connector_config)

                elif 'blip2' in self.config.model_name.lower() :
                    peft_model = get_peft_model(self.model.model.opt_model.model, connector_config)

                peft_model.delete_adapter("default")
                peft_model = peft_model.cpu()
                peft_model.save_pretrained(result_dir)
                # 저장 후 메모리 해제
                del peft_model

                torch.cuda.empty_cache()
                print("LoRA + Connector 모델 저장 완료(Gap 0 with train_compositional_edit.json) ->", result_dir)
            except:
                print("LoRA + Connector 모델 저장 실패")

        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats
    

        ## TEST - compositonal - two      
    def test_sequencial_compositional_connector_attention_rag_50(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True) # LLAVA:? / BLIP2: FT

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") # 궁금한게, 200개에 대한 batch
        averager = RunningStatAverager("val")

        start_time = time.time()
        ## 저장할 내용
        val_data_store = []

        # visul-data
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        # textual-data
        base_logits_store_tex = []

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 1. Inference Output for test locality(visual & textual 데이터/출력 저장 & 출력)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                # 1.1) visual edit part
                val_data_store.append(batch) # batch 데이터 저장
                with torch.no_grad():
                    base_outputs = self.model(batch["visual_edit"]["loc"]) # T-Loc inference 저장 # self.model -> ft
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["visual_edit"]["loc_image"]) # I-Loc inference 저장
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach())

                # 1.2) textual edit part
                with torch.no_grad():
                    base_outputs = self.model(batch["textual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_tex.append(base_logits.clone().detach())

                pbar.update(1)
            else:
                break
        pbar.close()

        ## 2. Model edit & Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            # 2.1) Model Edit (Update for a batch)
            # 2.1.1) Visual Edit(first)
            self.model.model.set_adapter("visual") # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["visual_edit"]["edit_inner"], mode = "visual" , peft = True)

            # 2.1.2) Textual Edit(second) 
            self.model.model.set_adapter("textual")  # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["textual_edit"]["edit_inner"], mode = "textual" , peft = True)

            # 2.1.3) Compositional Edit(second) 
            if val_step > 5:
                edited_model.model.set_adapter(["textual","visual","connector"])
                edited_model, _ = edited_model.edit(batch["port"][0], connector_mode=True) 


            # 2.2) Test with GAP
            if val_step >= gap_num: 
                # 기존 저장했던 batch, t-loc & i-loc-logit 불러옴. For Test
                stored_batch = val_data_store.pop(0) # vis + text
                stored_base_logits_vis = base_logits_store_vis.pop(0)
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0)
                stored_base_logits_tex = base_logits_store_tex.pop(0)

                # Test Sequential Edit(only inference & test) - vis / text 모두 다 평가해야 함.
                info_dict = self.test_sequencial_compositional_connector_attention_rag_step(
                    stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_tex
                    )
                averager.add(info_dict)

            # logging?
            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log_CompositionalEdit( ## ★☆★ 수정 필요 ★☆★ ##
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Logging Results ## 
        if log:
            self._inline_seq_log_CompositionalEdit(val_step, averager.average(), start_time, steps) ## ★☆★ 수정 필요 ★☆★ ##
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        if 'llava' in self.config.model_name.lower():
            if os.path.basename(self.config.name) == "llava-v1.5-13b":
                print("-> llava 13B 저장중...")
                result_dir = f"results/results_sequencial/llava1.5v_13b/composition/two_lora_connect_attention_rag_50"
            elif os.path.basename(self.config.name) == "llava-v1.5-7b":
                result_dir = f"results/results_sequencial/composition/two_lora_connect_attention_rag_50"
                print("-> llava 7B 저장중...")
            
        elif 'blip2' in self.config.model_name.lower() :
            result_dir = f"results/results_sequencial/blip2/composition/two_lora_connect_attention_rag_50"
        elif 'minigpt4' in self.config.model_name.lower() :
            result_dir = f"results/results_sequencial/minigpt4/composition/two_lora_connect_attention_rag_50"
        
        results_path = os.path.join(result_dir, f"{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json")
        
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        if gap_num == 0:
            try: # lora weight 저장
                from peft import LoraConfig, TaskType, get_peft_model, PeftConfig, PeftModel
                connector_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=8,
                        lora_alpha=16,
                        lora_dropout=0.05,
                        target_modules=["q_proj", "k_proj"]
                    )
                
                
                # blip이면
                if 'llava' in self.config.model_name.lower() :
                    peft_model = get_peft_model(self.model.model.base_model.model, connector_config)

                elif 'blip2' in self.config.model_name.lower() :
                    peft_model = get_peft_model(self.model.model.opt_model.model, connector_config)

                elif 'minigpt4' in self.config.model_name.lower() :
                    peft_model = get_peft_model(self.model.model.llama_model.model, connector_config)

                peft_model.delete_adapter("default")
                peft_model = peft_model.cpu()
                peft_model.save_pretrained(result_dir)
                # 저장 후 메모리 해제
                del peft_model

                torch.cuda.empty_cache()
                print("LoRA + Connector 모델 저장 완료(Gap 0 with train_compositional_edit.json) ->", result_dir)
            except:
                print("LoRA + Connector 모델 저장 실패")

        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats
    
    #     ## TEST - compositonal - two      
    def test_sequencial_compositional_connector_attention_rag_step(self, batch, edited_model, base_logits_vis, base_image_logits_vis, base_logits_tex):
        info_dict = {}

        ##############################################################################
        # ----------------------------Test: Visual Edit------------------------------#
        with torch.no_grad():
            # set lora: visual inference
            edited_model.model.set_adapter("visual")
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["visual_edit"]["edit_inner"])
            inner_batch_labels = batch["visual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels) # edit_loss_fn이 어딨지?
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(T-Gen)
            post_edit_outputs = edited_model(batch["visual_edit"]["edit_outer"])
            post_batch_labels = batch["visual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels) # edit_loss_fn -> vis, text 한번에 적용해도 되는가
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # image rephrase(I-Gen)
            post_image_edit_outputs = edited_model(batch["visual_edit"]["edit_outer_image"])
            post_image_batch_labels = batch["visual_edit"]["edit_outer_image"]["labels"]
            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
            else:
                post_image_edit_logits = post_image_edit_outputs

            if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)
            else:
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])
            del post_image_edit_outputs, post_image_edit_logits
            torch.cuda.empty_cache()

            # text loc(T-Loc)
            post_base_outputs = edited_model(batch["visual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_vis, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

            # image loc(I-Loc)
            post_image_base_outputs = edited_model(batch["visual_edit"]["loc_image"])
            if not isinstance(post_image_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
            else:
                post_image_base_logits = post_image_base_outputs
            post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
            base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits_vis, dim=-1), k=10, dim=-1).indices
            del post_image_base_outputs, post_image_base_logits
            torch.cuda.empty_cache()

        info_dict['vis/inner/acc'] = inner_edit_dict["acc"].item() # copy안해도 되는가? -> item은 int/float이 직접 반환됨. 따라서 ㅇㅇ
        info_dict['vis/edit/acc'] = post_edit_dict["acc"].item()
        info_dict['vis/image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["vis/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["vis/image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ##############################################################################
        # ----------------------------Test: Textual Edit------------------------------#
        with torch.no_grad():
            # set lora: visual inference
            edited_model.model.set_adapter("textual")
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["textual_edit"]["edit_inner"])
            inner_batch_labels = batch["textual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(Generality)
            post_edit_outputs = edited_model(batch["textual_edit"]["edit_outer"])
            post_batch_labels = batch["textual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # text loc(Locality)
            post_base_outputs = edited_model(batch["textual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_tex, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

        info_dict['text/inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['text/edit/acc'] = post_edit_dict["acc"].item()
        info_dict["text/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ################ portability #################
        # set lora: visual&textual inference
        edited_model.model.set_adapter(["textual","visual","connector"])

        if batch['port'] is not None:
            assert len(batch['port']) == 1, "batch['port'] exist and have only one element"
            port = batch['port'][0]
            with torch.no_grad():
                port_outputs = edited_model(port)
                port_labels = port["labels"]
                if not isinstance(port_outputs, torch.Tensor):
                    port_logits = port_outputs.logits
                else:
                    port_logits = port_outputs
                if port_logits.shape[1] > port_labels.shape[1]:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
                else:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
                port_acc = port_dict["acc"].item()
                del port_outputs, port_logits
                torch.cuda.empty_cache()

            info_dict['port/acc'] = port_acc
            ################ portability #################

        return info_dict

    # TEST - compositonal - two lora + Connector(att) - eval
    def test_sequencial_compositional_connector_attention_eval(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...") # 궁금한게, 200개에 대한 batch
        averager = RunningStatAverager("val")

        start_time = time.time()
        ## 저장할 내용
        val_data_store = []

        # visul-data
        base_logits_store_vis = []
        base_image_logits_store_vis = []
        # textual-data
        base_logits_store_tex = []

        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        
        ## 1. Inference Output for test locality(visual & textual 데이터/출력 저장 & 출력)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                # 1.1) visual edit part
                val_data_store.append(batch) # batch 데이터 저장
                with torch.no_grad():
                    base_outputs = self.model(batch["visual_edit"]["loc"]) # T-Loc inference 저장 # self.model -> ft
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_vis.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["visual_edit"]["loc_image"]) # I-Loc inference 저장
                    if not isinstance(base_image_outputs, torch.Tensor):
                            base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store_vis.append(base_image_logits.clone().detach())

                # 1.2) textual edit part
                with torch.no_grad():
                    base_outputs = self.model(batch["textual_edit"]["loc"]) # T-Loc inference 저장
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store_tex.append(base_logits.clone().detach())

                pbar.update(1)
            else:
                break
        pbar.close()
                                                    
        ## 2. Model edit & Test ##
        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            # 2.1) Model Edit (Update for a batch)
            # 2.1.1) Visual Edit(first)
            self.model.model.set_adapter("visual") # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["visual_edit"]["edit_inner"], mode = "visual" , peft = True)

            # 2.1.2) Textual Edit(second) 
            self.model.model.set_adapter("textual")  # PEFT -> set_adapter
            edited_model, _ = edited_model.edit(batch["textual_edit"]["edit_inner"], mode = "textual" , peft = True)

            # 2.2) Test with GAP
            if val_step >= gap_num: 
                # 기존 저장했던 batch, t-loc & i-loc-logit 불러옴. For Test
                stored_batch = val_data_store.pop(0) # vis + text
                stored_base_logits_vis = base_logits_store_vis.pop(0)
                stored_base_image_logits_vis = base_image_logits_store_vis.pop(0)
                stored_base_logits_tex = base_logits_store_tex.pop(0)

                # Test Sequential Edit(only inference & test) - vis / text 모두 다 평가해야 함.
                info_dict = self.test_sequencial_compositional_connector_attention_eval_step(
                    stored_batch, edited_model, stored_base_logits_vis, stored_base_image_logits_vis, stored_base_logits_tex
                    )
                averager.add(info_dict)

            # logging?
            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log_CompositionalEdit( ## ★☆★ 수정 필요 ★☆★ ##
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        ## Logging Results ## 
        if log:
            self._inline_seq_log_CompositionalEdit(val_step, averager.average(), start_time, steps) ## ★☆★ 수정 필요 ★☆★ ##
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        results_path = f"results/results_sequencial/composition/two_lora_connect_attention/eval/{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}_testnum{test_num}.json"
        
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats
    

        ## TEST - compositonal - two
    def test_sequencial_compositional_connector_attention_eval_step(self, batch, edited_model, base_logits_vis, base_image_logits_vis, base_logits_tex):
        info_dict = {}

        ##############################################################################
        # ----------------------------Test: Visual Edit------------------------------#
        with torch.no_grad():
            # set lora: visual inference
            edited_model.model.set_adapter("visual")
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["visual_edit"]["edit_inner"])
            inner_batch_labels = batch["visual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels) # edit_loss_fn이 어딨지?
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(T-Gen)
            post_edit_outputs = edited_model(batch["visual_edit"]["edit_outer"])
            post_batch_labels = batch["visual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels) # edit_loss_fn -> vis, text 한번에 적용해도 되는가
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # image rephrase(I-Gen)
            post_image_edit_outputs = edited_model(batch["visual_edit"]["edit_outer_image"])
            post_image_batch_labels = batch["visual_edit"]["edit_outer_image"]["labels"]
            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
            else:
                post_image_edit_logits = post_image_edit_outputs

            if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)
            else:
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])
            del post_image_edit_outputs, post_image_edit_logits
            torch.cuda.empty_cache()

            # text loc(T-Loc)
            post_base_outputs = edited_model(batch["visual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_vis, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

            # image loc(I-Loc)
            post_image_base_outputs = edited_model(batch["visual_edit"]["loc_image"])
            if not isinstance(post_image_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
            else:
                post_image_base_logits = post_image_base_outputs
            post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
            base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits_vis, dim=-1), k=10, dim=-1).indices
            del post_image_base_outputs, post_image_base_logits
            torch.cuda.empty_cache()

        info_dict['vis/inner/acc'] = inner_edit_dict["acc"].item() # copy안해도 되는가? -> item은 int/float이 직접 반환됨. 따라서 ㅇㅇ
        info_dict['vis/edit/acc'] = post_edit_dict["acc"].item()
        info_dict['vis/image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["vis/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["vis/image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ##############################################################################
        # ----------------------------Test: Textual Edit------------------------------#
        with torch.no_grad():
            # set lora: visual inference
            edited_model.model.set_adapter("textual")
            # inner(Reliability)
            inner_edit_outputs = edited_model(batch["textual_edit"]["edit_inner"])
            inner_batch_labels = batch["textual_edit"]["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase(Generality)
            post_edit_outputs = edited_model(batch["textual_edit"]["edit_outer"])
            post_batch_labels = batch["textual_edit"]["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # text loc(Locality)
            post_base_outputs = edited_model(batch["textual_edit"]["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits_tex, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

        info_dict['text/inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['text/edit/acc'] = post_edit_dict["acc"].item()
        info_dict["text/loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ################ portability #################
        # set lora: visual&textual inference
        edited_model.model.set_adapter(["textual","visual","connector"])

        if batch['port'] is not None:
            assert len(batch['port']) == 1, "batch['port'] exist and have only one element"
            port = batch['port'][0]
            with torch.no_grad():
                port_outputs = edited_model(port)
                port_labels = port["labels"]
                if not isinstance(port_outputs, torch.Tensor):
                    port_logits = port_outputs.logits
                else:
                    port_logits = port_outputs
                if port_logits.shape[1] > port_labels.shape[1]:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
                else:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
                port_acc = port_dict["acc"].item()
                del port_outputs, port_logits
                torch.cuda.empty_cache()

            info_dict['port/acc'] = port_acc
            ################ portability #################

        return info_dict




def set_edit_mode(model: torch.nn.Module, mode: str):
    """
    모델 내부의 모든 커스텀 어댑터(LoRA, MLP 연결 모듈)에 대해 편집 모드를 설정합니다.
    mode 인자는 "visual", "textual", "fusion", "default" 중 하나를 사용합니다.
    모듈이 해당 모드를 위한 메서드(use_vis_adapter, use_text_adapter, use_connector, set_default)를
    갖고 있다면 이를 호출합니다.
    """
    for module in model.modules():
        # fusion 모드: MLP 후처리를 활성화하는 경우
        if mode == "fusion":
            if hasattr(module, "use_connector"):
                module.use_connector()
        # visual 모드: visual 전용 adapter 활성화
        elif mode == "visual":
            if hasattr(module, "use_vis_adapter"):
                module.use_vis_adapter()
            elif hasattr(module, "set_default"):
                module.set_default()  # visual 전용 메서드가 없다면 기본 상태로 설정
        # textual 모드: textual 전용 adapter 활성화
        elif mode == "textual":
            if hasattr(module, "use_text_adapter"):
                module.use_text_adapter()
            elif hasattr(module, "set_default"):
                module.set_default()  # textual 전용 메서드가 없다면 기본 상태로 설정
        # default 모드: adapter를 비활성화(기본 상태)
        elif mode == "default":
            if hasattr(module, "set_default"):
                module.set_default()
        else:
            raise ValueError("Unsupported mode. Use 'visual', 'textual', 'fusion', or 'default'.")
