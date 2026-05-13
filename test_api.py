import base64
import json
import sys
from pathlib import Path

import requests


# API_URL = "http://127.0.0.1:8000/predict_font/"  # Local
API_URL = "https://unrepressed-bernardo-octagonal.ngrok-free.dev/predict_font/"  # Ngrok (cần header)


def call_api_with_url(image_url: str):
    """
    Gửi request tới API bằng image_url.
    """
    payload = {"image_url": image_url}
    print(f"Sending POST to {API_URL} with image_url={image_url!r}")
    headers = {}
    if "ngrok" in API_URL:
        headers["ngrok-skip-browser-warning"] = "true"
    response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
    print(f"Status code: {response.status_code}")
    try:
        data = response.json()
    except Exception:
        print("Không parse được JSON, raw response:")
        print(response.text)
        return

    # Lưu file ZIP về máy client nếu có
    if 'zip_base64' in data and data['zip_base64'] and 'zip_filename' in data and data['zip_filename']:
        try:
            # Decode base64
            zip_bytes = base64.b64decode(data['zip_base64'])
            
            # Lưu file vào thư mục hiện tại
            output_dir = Path.cwd() / "results"
            output_dir.mkdir(exist_ok=True)
            zip_path = output_dir / data['zip_filename']
            
            with open(zip_path, 'wb') as f:
                f.write(zip_bytes)
            
            print(f"\n Đã lưu file ZIP về: {zip_path.absolute()}")
            print(f"   Kích thước: {len(zip_bytes)} bytes")
        except Exception as e:
            print(f"\n  Lỗi khi lưu file ZIP: {e}")
    
    # Truncate zip_base64 để không spam terminal khi print
    if 'zip_base64' in data and data['zip_base64']:
        zip_len = len(data['zip_base64'])
        data['zip_base64'] = f"<BASE64_ENCODED_ZIP ({zip_len} chars)>"
    
    print("\n" + "="*60)
    print("RESPONSE JSON:")
    print("="*60)
    print(json.dumps(data, indent=2, ensure_ascii=False))


def image_file_to_base64(image_path: Path) -> str:
    """
    Đọc file ảnh và convert sang chuỗi base64 (không kèm prefix data:image/...).
    """
    with image_path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return encoded


def call_api_with_file(image_path: str):
    """
    Gửi request tới API bằng ảnh local (convert sang base64).
    """
    p = Path(image_path)
    if not p.is_file():
        print(f"File không tồn tại: {p}")
        return

    b64_str = image_file_to_base64(p)
    payload = {"image_base64": b64_str}

    print(f"Sending POST to {API_URL} with local file={str(p)!r}")
    headers = {}
    if "ngrok" in API_URL:
        headers["ngrok-skip-browser-warning"] = "true"
    response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
    print(f"Status code: {response.status_code}")
    try:
        data = response.json()
    except Exception:
        print("Không parse được JSON, raw response:")
        print(response.text)
        return

    # Lưu file ZIP về máy client nếu có
    if 'zip_base64' in data and data['zip_base64'] and 'zip_filename' in data and data['zip_filename']:
        try:
            # Decode base64
            zip_bytes = base64.b64decode(data['zip_base64'])
            
            # Lưu file vào thư mục hiện tại
            output_dir = Path.cwd() / "downloads"
            output_dir.mkdir(exist_ok=True)
            zip_path = output_dir / data['zip_filename']
            
            with open(zip_path, 'wb') as f:
                f.write(zip_bytes)
            
            print(f"\n✅ Đã lưu file ZIP về: {zip_path.absolute()}")
            print(f"   Kích thước: {len(zip_bytes)} bytes")
        except Exception as e:
            print(f"\n⚠️  Lỗi khi lưu file ZIP: {e}")
    
    # Truncate zip_base64 để không spam terminal khi print
    if 'zip_base64' in data and data['zip_base64']:
        zip_len = len(data['zip_base64'])
        data['zip_base64'] = f"<BASE64_ENCODED_ZIP ({zip_len} chars)>"
    
    print("\n" + "="*60)
    print("RESPONSE JSON:")
    print("="*60)
    print(json.dumps(data, indent=2, ensure_ascii=False))


def main():
    """
    Cách dùng:
    - Test với URL ảnh:
        python test_api.py url https://example.com/image.jpg

    - Test với file ảnh local:
        python test_api.py file /path/to/image.jpg
    """
    if len(sys.argv) < 3:
        print(main.__doc__)
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "url":
        image_url = sys.argv[2]
        call_api_with_url(image_url)
    elif mode == "file":
        image_path = sys.argv[2]
        call_api_with_file(image_path)
    else:
        print("Mode không hợp lệ. Dùng 'url' hoặc 'file'.")
        print(main.__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()


