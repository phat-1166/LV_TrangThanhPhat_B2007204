# Luận văn tốt nghiệp: Ứng dụng học sâu trong Phát hiện thận và Phân đoạn sỏi thận trên ảnh CT vùng thận

1. Cài đặt thư viện Streamlit: pip install streamlit
2. Cài đặt thư viện Aspose Imaging: pip install aspose-imaging-python-net
3. Clone github detectron2 và cài đặt framework Detectron2:
        git clone https://github.com/facebookresearch/detectron2.git
        python -m pip install -e detectron2
3. (Vào file demo.py) Chỉnh sửa đường dẫn đến file cấu hình gốc của model trong framework Detectron2 (trong github detectron2)
4. Chạy chương trình: python -m streamlit run demo.py