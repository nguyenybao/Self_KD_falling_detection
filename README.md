# Cải tiến mô hình học sâu cho bài toán nhận dạng hành động té ngã để hỗ trợ theo dõi người cao tuổi

Sinh viên thực hiện Mai Nguyen, Bao Yen Nguyen dưới sự hướng dẫn của cô Nguyễn Thị Thu Hiền và Thầy Vũ Đức Quang

## Tổng quan đề tài 
Tự chắt lọc tri thức, ý tưởng liên quan đến chắt lọc tri thức, là một cách tiếp cận mới lạ nhằm tránh đào tạo một mạng lưới giáo viên lớn.
Trong công việc của mình, chúng tôi đề xuất một cách tiếp cận chắt lọc kiến thức hiệu quả để nhận dạng té ngã. Theo cách tiếp cận của chúng tôi, mạng chia sẻ và học hỏi kiến thức được chắt lọc qua các vectơ được nhúng từ hai chế độ xem khác nhau của một điểm dữ liệu. Hơn nữa, chúng tôi cũng giới thiệu mạng nhẹ nhưng mạnh mẽ để giải quyết nhiệm vụ này dựa trên tích chập (2 + 1) D. Mạng được đề xuất của chúng tôi chỉ sử dụng tham số 0,1M giảm hàng trăm lần so với các mạng sâu khác. Để đánh giá hiệu quả của cách tiếp cận được đề xuất của chúng tôi, hai bộ dữ liệu tiêu chuẩn như bộ dữ liệu FDD và URFD, đã được thử nghiệm. Kết quả cho thấy hiệu suất hiện đại và vượt trội hơn so với đào tạo độc lập. Hơn nữa, với thông số 0,1M, mạng của chúng tôi cho thấy việc triển khai dễ dàng trên các thiết bị cạnh, ví dụ: điện thoại và máy ảnh, trong thời gian thực mà không cần GPU.


Hình dướng đây cho thấy cách tiếp cận của chúng tôi
<p align="center">
  <img width="800" alt="fig_method" src="https://github.com/vdquang1991/self_KD_falling_detection/blob/main/model.jpg">
</p>


## Chạy code

### Các thư viện cần thiết
- Python3
- Tensorflow (>=2.3.0)
- Numpy 
- Pillow
- Open-CV
- Scikit-learn
- ...
### Đào tạo trên 2 bộ dữ liệu là FDD và URFD

Trong code này, bạn có thể chạy kết quả thử nghiệm của nhiệm vụ nhận dạng té ngã. Bộ dữ liệu FDD và URFD được sử dụng trong giai đoạn đào tạo. Cài đặt đào tạo mẫu dành cho mạng được đề xuất của chúng tôi. Cài đặt siêu tham số chi tiết được liệt kê trong bài báo.

- Đào tạo với mô hình Self-KD
~~~
python train_tensor2.py --gpu=0 ----clip_len=16 --crop_size=224 --alpha=0.1 --use_mse=1 --lr=0.01 --drop_rate=0.5 --reg_factor=5e-4 --epochs=300 
~~~

tại, 

`--clip_len` is the length of the input video clips (number of frames).

`--crop_size`is the height and width size of all frames.

`--alpha` is the loss weight for the self-distillation loss (`=lambda` in the paper).

`--use_mse=1` if using our Self-KD approach otherwise `--use_mse=0`

`--lr` is learning rate init.

`--drop_rate` is drop rate of the dropout layer.

`--reg_factor` denote weight decay.

`--epochs` is the number of epochs for training. 

### Tập dữ liệu
Trong công việc này, chúng tôi đã sử dụng hai bộ dữ liệu tiêu chuẩn bao gồm FDD và URFD, để đánh giá hiệu suất của SKD so với các phương pháp hiện đại. Tất cả các video trong mỗi tập dữ liệu được trích xuất thành các khung trước khi đào tạo. Để trích xuất khung hình từ video, bạn có thể sử dụng lệnh `ffmpeg`.

Ví dụ như:

~~~
ffmpeg -i video_path.avi  "-vf", "fps=25" folder_path/%05d.jpg
~~~

Để tạo .csv files, please see `def split_train_test_data` in `gen_tensor2.py`. Cấu trúc của tệp csv như sau:
~~~
<FOLDER>,<#START_FRAMES>,<#END_FRAMES>,<LABEL>
~~~
Ví dụ:
~~~
Coffee_room/Videos/video_38,288,304,0
Home/Videos/video_57,16,32,0
Coffee_room/Videos/video_54,160,176,1
Coffee_room/Videos/video_15,0,16,0
Coffee_room/Videos/video_11,128,144,0
~~~



