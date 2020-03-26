
Project này bao gồm những mô hình cơ bản của kỹ thuật học sâu (Deep Learning):
# 1. Mạng nơ-ron:
Mạng nơ-ron đơn giản nhất ta có thể thấy là mạng nơ-ron toàn phần (Fully Connected Network)

<img width="477" alt="NN" src="https://user-images.githubusercontent.com/51883796/77663506-b5e70e80-6faf-11ea-963e-2703dfa237c6.PNG">
Tham khảo từ blog https://nttuan8.com/bai-4-backpropagation/
Trong đó: 
-Layer đầu tiên là input layer

-Các layer ở giữa được gọi là hidden layer

-Layer cuối cùng được gọi là output layer.

-Các hình tròn được gọi là node.

Mỗi mô hình luôn có 1 input layer, 1 output layer, có thể có hoặc không các hidden layer. Tổng số
layer trong mô hình được quy ước là số layer - 1 (Không tính input layer).

* Ví dụ như ở hình trên có 1 input layer, 2 hidden layer và 1 output layer. Số lượng layer của
mô hình là 3 layer.

Mỗi node trong hidden layer và output layer :

• Liên kết với tất cả các node ở layer trước đó với các hệ số w riêng.

• Mỗi node có 1 hệ số bias b riêng.

• Diễn ra 2 bước: tính tổng linear và áp dụng activation function.

Trong project này, ta áp dụng hàm activation là hàm sigmoid function

# 2. Phép toán truyền ngược (Backpropagation)

Truyền ngược (hay còn gọi là lan truyền ngược, Tiếng Anh: back-propagation), là một từ viết tắt cho "backward propagation of errors" tức là "truyền ngược của sai số", là một phương pháp phổ biến để huấn luyện các mạng thần kinh nhân tạo được sử dụng kết hợp với một phương pháp tối ưu hóa như gradient descent. Phương pháp này tính toán gradient của hàm mất mát với tất cả các trọng số có liên quan trong mạng nơ ron đó. Gradient này được đưa vào phương pháp tối ưu hóa, sử dụng nó để cập nhật các trọng số, để cực tiểu hóa hàm tổn thất.

Thuật toán Backpropagation:
![NN2](https://user-images.githubusercontent.com/51883796/77666853-0ceee280-6fb4-11ea-9c54-427067a73953.jpg)

Hình được lấy từ blog https://machinelearningcoban.com/2017/02/24/mlp/

# 3. Mạng nơ-ron tích chập (Convolutional Neural Network)

Giả sử ta có 1 ảnh màu kích thước 64x64. Trong xử lý ảnh, ảnh này được biểu diễn dưới dạng 1 tensor 64x64x3 (3 là 3 kênh màu nếu ta xét theo hệ màu RGB phổ biến hiện nay). Nên để biểu thị hết nội dung của bức ảnh thì cần truyền vào input layer tất cả các pixel (64x64x3 = 12288). Nghĩa là input layer giờ có 12288 nodes.

Giả sử số lượng node trong hidden layer 1 là 1000. Số lượng weight W (đường nối) giữa input layer và hiddenlayer 1 là 12288x1000 = 12288000, số lượng bias là 1000 => tổng số parameter là: 12289000. Đấy mới chỉ là số parameter giữa input layer và hidden layer 1, trong model còn nhiều layer nữa, và nếu kích thước ảnh tăng, ví dụ 512x512 thì số lượng parameter tăng cực kì nhanh => Cần giải pháp tốt hơn !!!

Vì vậy CNN ra đời. 

Đầu tiên ta sẽ tìm hiểu phép toán Convolution (phép tính tích chập)








