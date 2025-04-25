import matplotlib.pyplot as plt
import numpy as np

# 데이터 입력
sequential_gap = np.array([0, 10, 20, 50, 100])
dual_lora_int = np.array([45.51, 43.90, 43.93, 44.54, 42.14])
base_rag = np.array([63.70] * len(sequential_gap))  # 모든 Gap에서 동일한 값 유지
dual_lora_rag = np.array([69.91, 61.53, 61.34, 60.92, 60.45])
dual_lora_rag_conn = np.array([98.17, 96.04, 95.32, 95.09, 94.30])

single_lora_rag_conn = np.array(62.12, 47.33, 44.01, 42.61,40.22)


# X축을 동일한 간격으로 조정
sequential_gap_index = np.arange(len(sequential_gap))

# 그래프 생성
fig, ax = plt.subplots(figsize=(8, 5))

# 각 메서드의 성능 변화를 선 그래프로 표현
ax.plot(sequential_gap_index, dual_lora_int, marker='o', linestyle='-', linewidth=2, markersize=8, label="Dual-LoRA (Int.)", color='#1f77b4', alpha=0.9)
ax.fill_between(sequential_gap_index, dual_lora_int - 1, dual_lora_int + 1, color='#1f77b4', alpha=0.2)

ax.plot(sequential_gap_index, dual_lora_rag, marker='s', linestyle='-', linewidth=2, markersize=8, label="Dual-LoRA + RAG", color='#2ca02c', alpha=0.9)
ax.fill_between(sequential_gap_index, dual_lora_rag - 1, dual_lora_rag + 1, color='#2ca02c', alpha=0.2)

ax.plot(sequential_gap_index, dual_lora_rag_conn, marker='^', linestyle='-', linewidth=3.5, markersize=10, label="Dual-LoRA + RAG (w/ Conn.)", color='#d62728', alpha=0.9)
ax.fill_between(sequential_gap_index, dual_lora_rag_conn - 1, dual_lora_rag_conn + 1, color='#d62728', alpha=0.2)

# Base + RAG는 모든 Gap에서 같은 값이므로 점선으로 표현
ax.plot(sequential_gap_index, base_rag, linestyle='--', linewidth=2.5, color='black', label="Base + RAG (Fixed)", alpha=0.8)

# 그래프 스타일 설정
ax.set_xlabel("Sequential Gap", fontsize=14, fontweight='bold')
ax.set_ylabel("Compositional Reliability (%)", fontsize=14, fontweight='bold')
ax.set_title("Performance Comparison of Different Methods", fontsize=16, fontweight='bold', pad=15)

# X축 간격 균등하게 설정
ax.set_xticks(sequential_gap_index)
ax.set_xticklabels(sequential_gap, fontsize=12)
ax.tick_params(axis='y', labelsize=12)

# Y축을 0부터 시작하도록 설정
ax.set_ylim(0, max(dual_lora_rag_conn) + 5)

# 범례 설정 (논문 스타일)
ax.legend(fontsize=12, loc='lower left', frameon=True, fancybox=True, shadow=True, borderpad=1)

# 격자 추가 (논문 스타일)
ax.grid(True, linestyle="--", alpha=0.6)

# 여백 조정
plt.tight_layout()

# 논문용 벡터 그래픽으로 저장 (PDF, EPS, SVG 중 선택)
fig.savefig("performance_comparison.pdf", format="pdf", bbox_inches='tight')
fig.savefig("performance_comparison.eps", format="eps", bbox_inches='tight')
fig.savefig("performance_comparison.svg", format="svg", bbox_inches='tight')

# 그래프 출력
plt.show()
