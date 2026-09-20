# Filtered Neo4j database

Mục tiêu là tạo một Neo4j instance mới chỉ chứa phần đồ thị cần xử lý, không
thay đổi database gốc.

## Dữ liệu được sao chép

- `Post` có `knowledge_processed = false` và `posted_at` thuộc hai ngày được
  chọn. Mặc định là ngày 18–19/09/2026 theo múi giờ `Asia/Ho_Chi_Minh`.
- `Source` nối đến các Post đã chọn bằng `PUBLISHED`.
- `Entity` loại `LOCATION` hoặc `ORGANIZATION` là một trong hai đầu của quan hệ
  `PART_OF`, `IN_REGION`, `JURISDICTION` hoặc `SUBORDINATE_TO`.
- Tất cả properties của node và tất cả relationship có hai đầu thuộc tập node
  đã chọn đều được giữ nguyên.

Entity không tham gia một trong bốn quan hệ phân cấp sẽ không được lấy. Việc
chọn Entity độc lập với Post vì các Post chưa xử lý thường chưa có `MENTIONS`.

## Địa chỉ hai Neo4j instance

| Instance | Browser | Bolt |
|---|---|---|
| Database gốc | `http://localhost:7474` | `bolt://localhost:7687` |
| Database rút gọn | `http://localhost:7475` | `bolt://localhost:7688` |

Hai instance sử dụng Docker volume riêng nên dữ liệu không bị trộn lẫn.

## Khởi động database phân tích

```bash
docker compose up -d neo4j_filtered
```

## Biến môi trường

Tất cả chương trình dùng chung cấu hình Neo4j trong `.env`:

```env
NEO4J_URI=bolt://localhost:7688
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

## Import Facebook vào database mới

Importer Facebook mặc định ghi vào `bolt://localhost:7688`:

```bash
python3 import_database/import_facebook_from_postgreSQL.py
```

Importer lấy địa chỉ từ `NEO4J_URI` trong `.env`.

## Chạy phân tích trên database mới

Script `scripts/extract_entities.py` cũng mặc định kết nối tới `bolt://localhost:7688`:

```bash
python -m scripts.extract_entities
```

`scripts/extract_entities.py` và `scripts/rerun_modules.py` cùng lấy địa chỉ từ `NEO4J_URI`
trong `.env`.

## Cấu trúc thư mục

- `backend/`: API và truy vấn Neo4j phục vụ tìm kiếm/chat.
- `knowledge_relations/`: logic quan hệ, phân cấp entity và hợp nhất event.
- `import_database/`: các chương trình nhập dữ liệu nguồn.
- `scripts/`: các entrypoint vận hành, backfill, migration và preview. Chạy từ thư mục gốc bằng `python -m scripts.<tên_script>`.
- `tests/`: toàn bộ automated test; chạy bằng `python -m pytest`.
- `docs/`: tài liệu thiết kế và kế hoạch kỹ thuật.
