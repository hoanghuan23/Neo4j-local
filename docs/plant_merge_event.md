Dưới đây là bản kế hoạch hoàn chỉnh, đã bổ sung xử lý thời gian, không giới hạn tổng số ứng viên, gộp nhiều Event và kiểm tra xung đột trước khi gộp. Bạn có thể dùng trực tiếp làm `plan_merge_event.md`.

Kế hoạch thay đổi logic gộp sự kiện (Event Consolidation)

# Kế hoạch thay đổi logic gộp sự kiện (Event Consolidation)

## Mục tiêu

Cải tiến quy trình gộp sự kiện nhằm:

* Tìm đầy đủ các Event có khả năng đề cập đến cùng một occurrence.

* Không bỏ sót ứng viên chỉ vì giới hạn top 10.

* Giảm chi phí AI bằng cách xếp hạng và xử lý theo batch.

* Cho phép gộp nhiều Event trong cùng một lượt xử lý, thay vì chỉ chọn Event có điểm cao nhất.

* Hạn chế gộp sai do AI trích xuất không chính xác thời gian, địa điểm hoặc chủ thể.

* Bảo toàn toàn bộ EventMention và quan hệ khi thực hiện merge.

Quy trình gồm 4 bước:

1. Candidate Retrieval.

2. Candidate Ranking & Batching.

3. AI Matching.

4. Evidence-based Merge Guard & Multi-event Merge.

---

# 1. Candidate Retrieval

## 1.1. Mục tiêu

Truy xuất toàn bộ Event ứng viên dựa trên hai thông tin chính:

* `location`: địa điểm xảy ra sự kiện.

* `occurrence_times`: thời gian xảy ra sự kiện.

Không sử dụng hành động, chủ thể hoặc đối tượng làm một nhánh tìm kiếm độc lập.

Các thông tin này sẽ được sử dụng ở bước Ranking.

## 1.3. Quy tắc truy xuất ứng viên

**Trường hợp 1: Có Location và occurrence_times**

* Location phải tương thích theo phân cấp địa lý.

* Khoảng thời gian xảy ra phải tương thích.

* Xét độ chính xác của thời gian khi so sánh.

**Trường hợp 2: Có Location nhưng không có occurrence_times**

* Lọc theo Location tương thích.

* Sử dụng `posted_at` của Post chứa EventMention làm mốc thời gian tìm kiếm.

* Khoảng thời gian tìm kiếm có thể cấu hình.

**Trường hợp 3: Không có Location nhưng có occurrence_times**

* Chỉ lọc theo khoảng thời gian xảy ra.

* Không loại bỏ Event vì thiếu Location.

**Trường hợp 4: Không có cả Location và occurrence_times**

* Lọc theo `posted_at` trong khoảng thời gian cấu hình.

Khi một bên có occurrence_times và bên còn lại không có, dùng thời gian đăng của bên thiếu để tìm kiếm với khoảng sai lệch cấu hình.

Khi cả hai đều thiếu occurrence_times, đối chiếu thời gian đăng của các Post.

Đối với Event đã có nhiều EventMention, sử dụng các mốc thời gian và bằng chứng từ các mention liên quan, tránh chỉ lấy một ngày đăng đại diện.

**Quy tắc quan trọng:**

* `posted_at` chỉ là tín hiệu truy xuất ứng viên.

* Không sử dụng `posted_at` fallback làm bằng chứng xác nhận hoặc bác bỏ hai occurrence.

* Location tương thích theo phân cấp, nhưng không coi mọi địa điểm cùng thuộc một tỉnh hoặc quốc gia là cùng địa điểm xảy ra.

* Thời gian có độ chính xác thấp không được xem như thời gian chính xác đến từng ngày.

* Cửa sổ thời gian chỉ giới hạn phạm vi truy xuất, không phải bằng chứng hai sự kiện khác nhau.

## 1.4. Kết quả

Trả về toàn bộ Event thỏa mãn điều kiện truy xuất.

Loại bỏ:

* Event hiện tại.

* Event trùng lặp trong danh sách candidate.

Không giới hạn top 10 tại bước này.

---

# 2. Candidate Ranking & Batching

## 2.1. Mục tiêu

Xếp hạng ứng viên để ưu tiên các Event có dấu hiệu trùng mạnh và tối ưu chi phí gọi AI.

Ranking không phải quyết định gộp sự kiện.

## 2.2. Tiêu chí xếp hạng

Tính điểm dựa trên:

* Hành động trung tâm.

* Chủ thể chính.

* Đối tượng hoặc nạn nhân.

* Loại sự kiện.

* Từ vựng và chi tiết đặc trưng.

Ưu tiên sự tương đồng về danh tính và các đặc điểm cụ thể hơn sự tương đồng từ vựng chung.

Thông tin chỉ xuất hiện ở một phía không được mặc định là mâu thuẫn.

## 2.3. Không giới hạn tổng số ứng viên

Không sử dụng giới hạn cứng:

`MAX_TOTAL_CANDIDATES = 10`

Nếu tìm được 35 Event đủ điều kiện, giữ cả 35 để xử lý.

Ranking chỉ quyết định thứ tự xử lý.

Không loại ứng viên chỉ vì nằm ngoài top 10.

## 2.4. Xử lý theo batch

Cấu hình:

`MAX_CANDIDATES_PER_BATCH = 10`

Ví dụ có 35 ứng viên:

* Batch 1: 10 Event.

* Batch 2: 10 Event.

* Batch 3: 10 Event.

* Batch 4: 5 Event.

AI xử lý lần lượt từng batch.

Không dừng khi batch đầu tiên đã tìm được SAME_EVENT.

Tiếp tục cho đến khi toàn bộ ứng viên đủ điều kiện được xử lý.

Nếu việc xử lý bị gián đoạn hoặc vượt giới hạn chi phí cấu hình, lưu trạng thái các ứng viên chưa xử lý để tiếp tục sau; không được âm thầm coi chúng là DIFFERENT_EVENT.

Mỗi candidate là một Event duy nhất, không đưa nhiều EventMention của cùng một Event thành các candidate riêng biệt.

---

# 3. AI Matching

## 3.1. Mục tiêu

AI so sánh EventMention mới với từng Event ứng viên và xác định chúng có cùng một occurrence cụ thể hay không.

## 3.2. Dữ liệu đầu vào

Cung cấp:

* Thông tin EventMention mới.

* Thông tin từng Event ứng viên.

* Mô tả và thuộc tính liên quan.

* Thời gian xảy ra đã chuẩn hóa và biểu thức gốc.

* Địa điểm và các thực thể liên quan.

* Bằng chứng từ những EventMention đã thuộc Event ứng viên khi cần thiết.

Không dùng riêng mô tả tổng hợp của Event nếu việc đó làm mất các chi tiết nhận diện quan trọng.

Phân biệt rõ thời gian xảy ra, thời gian đăng bài và thời gian cập nhật/thông báo.

## 3.3. Quy tắc so sánh

Đối chiếu:

* Hành động trung tâm.

* Chủ thể.

* Đối tượng hoặc nạn nhân.

* Thời gian xảy ra.

* Địa điểm xảy ra.

* Kết quả và số lượng.

* Chi tiết nhận diện đặc trưng.

Không mặc định hai sự kiện khác nhau chỉ vì:

* Một phía có ít thông tin hơn.

* Tên địa điểm khác cấp hành chính nhưng tương thích.

* Cách diễn đạt khác nhau.

* Loại sự kiện khác nhau nhưng vẫn mô tả cùng hành động.

* Ngày đăng bài khác nhau.

Ngược lại, cùng ngày, cùng địa điểm hoặc cùng chủ thể không đủ để xác nhận cùng occurrence.

## 3.4. Kết quả AI

AI phải trả về một quyết định cho mỗi `candidate_event_key`:

**SAME_EVENT**

Có đủ bằng chứng cho thấy hai bản ghi đề cập cùng một occurrence cụ thể.

**DIFFERENT_EVENT**

Có căn cứ xác định hai bản ghi đề cập những occurrence khác nhau.

**POSSIBLE_SAME_EVENT**

Có dấu hiệu trùng nhưng chưa đủ bằng chứng để kết luận.

Mỗi kết quả gồm:

* `candidate_event_key`.

* `decision`.

* `confidence`.

* `reasons`.

Phải có kết quả cho tất cả ứng viên trong batch.

Không chỉ trả về ứng viên có confidence cao nhất.

Không tự động lựa chọn duy nhất một SAME_EVENT nếu có nhiều ứng viên cùng đủ bằng chứng.

---

# 4. Evidence-based Merge Guard & Multi-event Merge

## 4.1. Mục tiêu

Backend kiểm tra toàn bộ ứng viên được AI quyết định SAME_EVENT.

Không chỉ kiểm tra hoặc gộp ứng viên có ranking cao nhất.

Sử dụng `evaluate_merge_guard()` để xác minh quyết định trước khi thực hiện merge.

## 4.2. Quy tắc kiểm tra

Guard kiểm tra:

**Hành động trung tâm**

BLOCK nếu xác nhận hai hành động trung tâm không thể thuộc cùng một occurrence.

**Chủ thể và đối tượng/nạn nhân**

BLOCK khi có mâu thuẫn danh tính rõ ràng, đáng tin cậy và không thể tương thích.

Thiếu thông tin không phải mâu thuẫn.

**Thời gian xảy ra**

Chỉ BLOCK khi có bằng chứng đáng tin cậy rằng hai thời gian xảy ra không thể tương thích.

Cần phân biệt ngày xảy ra với ngày đăng, thông báo hoặc cập nhật sự kiện.

Không dùng posted_at fallback để BLOCK.

Nếu dữ liệu thời gian có khả năng trích xuất sai hoặc chưa đủ chắc chắn, chuyển sang REVIEW thay vì BLOCK.

**Địa điểm**

Cho phép địa điểm tương thích theo phân cấp.

Không BLOCK chỉ vì hai tên địa điểm khác nhau.

Chỉ BLOCK khi xác định được hai địa điểm xảy ra thực sự loại trừ nhau và không thể thuộc cùng occurrence.

Trường hợp địa điểm khác nhau nhưng chưa xác minh được, chuyển REVIEW.

**Loại sự kiện**

Không BLOCK chỉ vì `type` khác nhau.

Kiểm tra hành động thực tế và bản chất occurrence.

**Confidence**

Kiểm tra ngưỡng mặc định:

`MERGE_CONFIDENCE_THRESHOLD = 0.90`

Confidence chỉ là điều kiện bổ sung, không thay thế bằng chứng.

## 4.3. Trạng thái Guard

**BLOCK**

Có mâu thuẫn rõ ràng, đáng tin cậy và không thể tương thích.

**REVIEW**

Có mâu thuẫn chưa xác minh được hoặc bằng chứng chưa đủ chắc chắn.

**PASS**

Không phát hiện mâu thuẫn đủ chắc chắn để chặn gộp.

PASS không tự nó chứng minh hai sự kiện trùng nhau; vẫn phải kết hợp quyết định SAME_EVENT của AI và ngưỡng confidence.

Khi có mâu thuẫn, guard phải ưu tiên kiểm tra dữ liệu chuẩn hóa và bằng chứng gốc, thay vì chỉ so sánh các thuộc tính AI đã trích xuất.

## 4.4. Quyết định cuối cùng

* BLOCK → DIFFERENT_EVENT.

* REVIEW → POSSIBLE_SAME_EVENT.

* PASS + AI SAME_EVENT + confidence đạt ngưỡng → SAME_EVENT.

Giữ lý do và kết quả guard để phục vụ kiểm tra lỗi.

## 4.5. Xử lý nhiều ứng viên SAME_EVENT

Sau khi đánh giá toàn bộ candidate, thu thập tất cả Event có quyết định cuối cùng là SAME_EVENT.

Không chỉ lấy Event có ranking hoặc confidence cao nhất.

Ví dụ:

* Event A → SAME_EVENT.

* Event B → SAME_EVENT.

* Event C → SAME_EVENT.

Cả ba đều được đưa vào danh sách xem xét gộp.

Tuy nhiên, cần kiểm tra tính nhất quán của nhóm trước khi thực hiện merge.

## 4.6. Kiểm tra xung đột giữa các ứng viên

Trước khi gộp nhiều Event vào một node:

* Kiểm tra các Event được chọn có mâu thuẫn với nhau không.

* Đối chiếu thời gian, địa điểm, hành động, chủ thể và đối tượng.

* Kiểm tra bằng chứng từ các EventMention đã gộp trước đó.

* Không suy ra B và C chắc chắn trùng nhau chỉ vì A được đánh giá trùng với cả B và C.

Nếu phát hiện mâu thuẫn:

* Không tự động gộp toàn bộ nhóm.

* Đưa những cặp có xung đột chưa xác minh được vào REVIEW.

* Chỉ gộp nhóm có bằng chứng xác nhận cùng occurrence và không tồn tại xung đột chưa giải quyết.

Việc kiểm tra chéo chỉ cần tập trung vào các nhóm có nhiều ứng viên SAME_EVENT hoặc xuất hiện dấu hiệu mâu thuẫn, tránh gọi thêm AI không cần thiết.

## 4.7. Thực hiện Multi-event Merge

Nếu nhiều Event vượt qua toàn bộ điều kiện:

* Chọn một Event làm node đại diện theo quy tắc ổn định.

* Gộp tất cả Event đủ điều kiện vào node đại diện.

* Chuyển toàn bộ EventMention về Event đại diện qua EVIDENCE_FOR.

* Bảo toàn các quan hệ và property cần thiết.

* Cập nhật `legacy_event_keys`.

* Cập nhật các thuộc tính tổng hợp từ bằng chứng đã xác minh.

* Không ghi đè dữ liệu đáng tin cậy bằng dữ liệu thiếu hoặc mâu thuẫn.

Thực hiện merge trong transaction để tránh dữ liệu chỉ được gộp một phần.

Đảm bảo chạy lại pipeline không tạo quan hệ trùng hoặc làm mất bằng chứng.

---

# 5. Các trường hợp cần kiểm thử

**Test 1: Nhiều bài cùng một sự kiện**

20 Post đề cập cùng một vụ tai nạn.

Kết quả mong muốn: một Event đại diện, bảo toàn 20 EventMention.

**Test 2: Vượt quá 10 ứng viên**

Candidate Retrieval tìm thấy 35 Event.

Kết quả mong muốn:

* Ranking toàn bộ 35.

* Xử lý đủ 4 batch.

* Không bỏ qua ứng viên ngoài top 10.

**Test 3: Thiếu thời gian xảy ra**

Một bài có occurrence_times, bài còn lại không có.

Kết quả mong muốn: posted_at được dùng để truy xuất, không được coi là bằng chứng mâu thuẫn.

**Test 4: Khác cấp địa điểm**

Một bài ghi Hà Nội, bài còn lại ghi Thanh Xuân.

Kết quả mong muốn: nhận diện địa điểm tương thích theo phân cấp, không BLOCK chỉ vì khác tên.

**Test 5: Một sự kiện có nhiều ứng viên trùng**

AI trả SAME_EVENT cho 3 Event.

Kết quả mong muốn: kiểm tra tính nhất quán và gộp tất cả Event thực sự thuộc cùng occurrence, không chỉ chọn ứng viên điểm cao nhất.

**Test 6: Xung đột giữa các ứng viên**

A giống B và A giống C, nhưng B và C có bằng chứng mâu thuẫn.

Kết quả mong muốn: không tự động gộp toàn bộ ba Event; giữ các trường hợp chưa xác minh ở trạng thái REVIEW.

**Test 7: Chạy lại pipeline**

Một nhóm Event đã được gộp trước đó.

Kết quả mong muốn: không tạo thêm Event hoặc quan hệ trùng, không làm mất EventMention.

---

# 6. Yêu cầu triển khai

* Tận dụng các hàm và schema hiện tại trong `event_hierarchy.py`.

* Giữ cấu trúc quyết định SAME_EVENT, DIFFERENT_EVENT, POSSIBLE_SAME_EVENT.

* Giữ cơ chế guard hiện có nhưng điều chỉnh điều kiện BLOCK/REVIEW/PASS theo bằng chứng.

* Thay giới hạn tổng 10 candidate bằng xử lý batch tối đa 10 ứng viên mỗi lượt.

* Không dừng xử lý sau khi tìm được SAME_EVENT đầu tiên.

* Bổ sung xử lý gộp nhiều Event và kiểm tra xung đột trong nhóm.

* Không thay đổi dữ liệu Event đã gộp trước đó; áp dụng logic mới cho các lượt xử lý tiếp theo.

* Bổ sung log về số candidate tìm được, số batch, số SAME_EVENT, số REVIEW, số BLOCK và số Event thực sự được gộp.

**Kết quả cuối cùng mong muốn:**

Candidate Retrieval → Ranking toàn bộ → AI Matching theo batch → Guard từng ứng viên → Kiểm tra tính nhất quán → Gộp toàn bộ Event đủ điều kiện.
