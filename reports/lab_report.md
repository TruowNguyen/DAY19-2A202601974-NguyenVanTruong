# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Nguyễn Văn Trường
**Khóa học:** AICB-K34 · Track 3: GraphRAG  
**Ngày thực hiện:** 20/08/2026
**Phạm vi:** 5.000 dòng đầu HackerNoon; 2.105 bản tin sau exact dedup; lấy mẫu 1.500 bản tin/chunk; tối đa 400 chunk cho extraction.
**Model cuối:** `qwen/qwen3.6-27b` cho generation và LLM Judge; checkpoint đủ 50/50 câu.

## 1. Thuyết minh kỹ thuật và phân tích ca lỗi

### 1.1 Coreference Resolution

Spot-check tại `chunk_id=5d0b9cbc04b00a5fe84a::c0000` có Walt Disney Co., CEO Bob Iger và các cụm “its streaming services”, “its ad-free Disney+ and Hulu plans”. Antecedent gần nhất có thể là Bob Iger, nhưng sở hữu thực tế thuộc Walt Disney Co. Cơ chế trong notebook chỉ thay đại từ khi antecedent được hỗ trợ rõ trong cùng chunk; nếu không chắc chắn thì giữ nguyên và ghi `unresolved_mentions`.

Nếu phân giải nhầm `its` thành Bob Iger, graph có thể tạo cạnh sai từ `Person` tới Disney+/Hulu hoặc gán quyết định tăng giá cho cá nhân thay vì công ty. Biện pháp giảm lỗi là giữ coreference bảo thủ, lưu chunk gốc làm provenance và spot-check batch có `COREF_BATCH_FAILED`.

### 1.2 Entity Resolution và Lexical Guard

- Ngưỡng merge vector: `threshold = 0.90`.
- Lexical guard: chỉ merge khi tên sau chuẩn hóa/hủy hậu tố giống nhau hoặc có `SequenceMatcher >= 0.72`.
- Audit có 45 cặp. Không có cặp không đồng nhất nào đạt cosine `> 0.85`; vì vậy không thể trích dẫn trung thực một `REJECT_GUARD > 0.85` trong sample.
- Cặp gần nhất quan sát được là `Artificial Intelligence` và `Machine Learning`, cosine `0.703462`, quyết định `REJECT_THRESHOLD`. Hai khái niệm liên quan nhưng không đồng nhất nên không merge.

Khi scale dữ liệu cần giữ log `MERGE_MANUAL`, `MERGE_VECTOR`, `REJECT_GUARD`, `REJECT_THRESHOLD` và review thủ công các cặp sát ngưỡng.

### 1.3 Graph và Super-node Mitigation

Graph cuối có **99 nodes**, **55 edges**, và **0 edge thiếu provenance**.

| Hạng | Thực thể | Type | Degree |
|---:|---|---|---:|
| 1 | DI | Company | 2 |
| 2 | Eric Schummer | Person | 2 |
| 3 | Aqara | Company | 2 |

Không node nào đạt `degree > 100`; cả 50 evaluation queries có `graph_supernode_events = 0`. Policy cap 50 cạnh tồn tại nhưng chưa được kích hoạt. Temporal cap giúp tránh context explosion và ưu tiên thông tin mới, nhưng có thể làm mất sự kiện lịch sử. Nên kết hợp relevance, time filter theo câu hỏi và diversity theo relation.

### 1.4 Benchmark Flat RAG và GraphRAG

| Metric | Flat RAG | GraphRAG | Δ Graph−Flat |
|---|---:|---:|---:|
| Comprehensiveness | 2.480 | 2.280 | -0.200 |
| Faithfulness | 4.940 | 4.860 | -0.080 |
| Multi-hop reasoning | 3.520 | 3.320 | -0.200 |
| Latency trung bình (s) | 5.064 | 3.420 | -1.644 |
| Token trung bình | 729.98 | 636.44 | -93.54 |

| Group | Số câu | Flat comp. | Graph comp. | Flat multi-hop | Graph multi-hop |
|---|---:|---:|---:|---:|---:|
| factoid | 5 | 2.600 | 2.600 | 5.000 | 5.000 |
| cross-doc | 22 | 2.636 | 2.500 | 3.364 | 3.227 |
| multi-hop | 23 | 2.304 | 2.000 | 3.348 | 3.043 |

Latency/token GraphRAG thấp hơn không chứng minh graph rẻ hơn: graph context thường rỗng/ngắn nên câu trả lời “insufficient evidence” ngắn hơn; thứ tự gọi API và rate-limit cũng làm latency Flat tăng.

`G5000-01` hỏi giao dịch Aeris–Ericsson. Cả hai pipeline trả thiếu bằng chứng vì chunk liên quan không nằm trong index mẫu. Judge vẫn chấm cao vì câu trả lời faithful với candidate context; đây là dấu hiệu prompt Judge chưa phạt đủ việc không khớp reference.

`G5000-26` là ca GraphRAG thất bại rõ: Flat tìm đúng `Cohere` và conversational customer-service agents, đạt 5/5/5. GraphRAG trả “insufficient evidence”, đạt 1/5/1. Nguyên nhân là graph extraction/retrieval không giữ hoặc nối đúng evidence dù vector context có chunk liên quan. `G5000-48` tương tự: Flat tổng hợp đủ ba record Snowflake, còn GraphRAG bỏ lỡ market recognition và chỉ đạt 2/5/2.

Khắc phục: tăng extraction coverage, lưu `Document/Chunk` nodes, seed bằng lexical + vector trước LLM, và luôn hợp nhất top vector evidence vào hybrid context.

### 1.5 Trade-offs, Agent Control và Scale 350 MB

- Flat RAG đơn giản, index nhanh và recall tốt cho câu hỏi gần từ khóa; hạn chế là khó nối quan hệ xuyên tài liệu.
- GraphRAG có provenance/traversal nhưng phụ thuộc mạnh vào extraction/entity resolution; graph thưa làm recall giảm.
- Từ chối pairwise cosine `O(N²)` trên toàn bộ 350 MB; dùng FAISS và blocking theo entity type.
- Bottleneck khi scale là LLM coreference/NER-RE, rồi entity resolution và ingestion. Giải pháp: async batch workers, checkpoint/idempotency, retry theo rate-limit, ANN/HNSW, bulk `UNWIND`, community partitioning và incremental update.

## 2. Reflection và Action Plan

### 2.1 Mapping bài giảng vào code

| Khái niệm | Hàm/khối code | Quan sát |
|---|---|---|
| Conservative Coreference | `resolve_coref_batch()`, `run_coref()` | Batch lỗi fallback text gốc và đánh dấu thất bại. |
| Schema/Allowlist Guard | `ALLOWED_*`, `dict_items()` | Chặn type/relation ngoài schema và JSON sai kiểu. |
| Bulk Cypher | `bulk_insert_nodes()`, `bulk_insert_edges()` | `UNWIND`; 0 edge thiếu provenance. |
| Entity Resolution | `build_resolution_map()`, `UF`, `merge_guard()` | Threshold 0,90; audit cả reject dưới ngưỡng. |
| Flat Retrieval | `build_flat_index()`, `retrieve_flat_context()` | FAISS FlatIP trên 1.500 chunks. |
| Hybrid Retrieval | `match_seeds()`, `retrieve_graph_context()` | Graph thưa; vector fallback là bắt buộc. |
| Super-node | `SUPER_NODE_DEGREE`, `SUPER_NODE_EDGE_CAP` | Có policy, sample chưa kích hoạt. |
| LLM Judge | `judge_pair()`, `run_evaluation()` | Gộp hai phương pháp/call, resume checkpoint đủ 50/50. |

### 2.2 Debugging và bài học

Chuỗi lỗi chính gồm model cũ 404, model 120B hết TPD 429, response JSON có `items` là string, và provider/model trong kernel lệch `.env`. Cách xử lý:

1. Chuyển sang `qwen/qwen3.6-27b`, tắt reasoning cho tác vụ thường.
2. Thêm `dict_items()` để bỏ phần tử sai kiểu thay vì crash.
3. Retry 429 theo thời gian `try again in ...`.
4. Gộp hai Judge calls thành `judge_pair()` để giảm token.
5. Resume theo `generation_model`, `judge_provider`, `judge_model` để không trộn kết quả.
6. Dùng `load_dotenv(..., override=True)` để tránh kernel giữ biến cũ.

Bài học: production RAG phải coi model availability, quota, malformed output, checkpoint và idempotency là yêu cầu kiến trúc.

### 2.3 Kế hoạch áp dụng

**Đồ án:** tra cứu tri thức doanh nghiệp/công nghệ từ tin tức. Hybrid RAG phù hợp hơn GraphRAG thuần vì phần lớn truy vấn cần recall văn bản, graph bổ sung multi-hop có bằng chứng.

- Nodes: `Company`, `Person`, `Technology`, mở rộng `Event`, `Document`, `Chunk`.
- Relations: `ACQUIRED`, `DEVELOPED`, `INVESTED_IN`, `FOUNDED`, `WORKED_AT`, `PARTNERED_WITH`, `USES`, `LEADS`, kèm provenance/date/confidence.
- Entity Resolution: blocking theo type/domain, ANN top-k, lexical guard, manual aliases và human review vùng sát ngưỡng.
- Super-node: cap theo relation, relevance/time-aware sampling và diversity theo nguồn.
- Evaluation: tách correctness so với reference khỏi faithfulness với context.

## 3. Tự đánh giá

| Tiêu chí | Điểm | Ghi chú |
|---|---:|---|
| Hiểu GraphRAG | 4/5 | Nắm hybrid retrieval, provenance và failure mode graph thưa. |
| Kiểm soát AI Agent | 4/5 | Có schema/parser guard, retry và checkpoint. |
| Chất lượng graph | 3/5 | Provenance đúng nhưng coverage thấp: 99 nodes/55 edges. |
| Debug hệ thống | 4/5 | Xử lý model, quota, JSON, env và resume. |

## 4. Deliverables và giới hạn

- `outputs/graphrag_eval_results.csv`: 50 dòng.
- `outputs/graphrag_vs_flatrag_summary.csv`: 15 dòng.
- `outputs/entity_resolution_audit.csv`: 45 dòng.
- `outputs/raw_triples.csv`, `canonical_triples.csv`, `top_degree_nodes.csv`.
- Neo4j: 99 nodes, 55 edges, 0 edge thiếu provenance.

Giới hạn: extraction coverage chưa đủ cho Golden Dataset, không có super-node thực, và Judge có xu hướng thưởng “insufficient evidence” nếu faithful với context dù không đúng reference. Đây là ưu tiên cải tiến tiếp theo.
