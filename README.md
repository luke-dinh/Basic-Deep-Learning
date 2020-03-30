
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

Trong xử lý ảnh, ta có khái niệm kernel. Kernel là một ma trận vuông kích cỡ KxK (thông thường ta chọn K là số lẻ, lí do sẽ đề cập ở phần sau). Kí hiệu phép tính convolution (⊗), kí hiệu Y = X ⊗ W . Kernel có vai trò tính tích chập trên từng tensor và tại mỗi vị trí pixel. 

Ta lấy ví dụ: 
<img width="431" alt="NN3" src="https://user-images.githubusercontent.com/51883796/77851457-787abf00-7203-11ea-9f9c-78519930ef8e.PNG">

Một cách tổng quát, Với mỗi phần tử xi j trong ma trận X lấy ra một ma trận có kích thước bằng kích thước của kernel W có phần tử x(i,j) làm trung tâm (đây là vì sao kích thước của kernel thường lẻ) gọi là ma trận A. Sau đó tính tổng các phần tử của phép tính element-wise của ma trận A và ma trận W, rồi viết vào ma trận kết quả Y. 

Trong ví dụ trên, vị trí Y(1,1) của Convolved Feature được tính bằng cách nhân từng phần tử của vùng ma trận kích thước 3x3 của ảnh được kernel đè lên với từng phần tử của kernel. 

Tuy nhiên thì sau khi tính tích chập, ta thấy kích cỡ của Convolved Feature bị giảm đi. Đó là bởi output của Convolved Feature sẽ trùng với center của kernel được đưa vào. Trong ví dụ trên, center của kernel được đặt tại tâm của kernel. Nếu như áp dụng công thức như trên thì vùng biên ảnh sẽ được xử lý như thế nào để output vẫn giữ nguyên kích thước ban đầu của ảnh? Phép toán Padding (mở rộng) ra đời. 

<img width="403" alt="NN4" src="https://user-images.githubusercontent.com/51883796/77851949-28512c00-7206-11ea-8b28-33150f0e69c1.PNG">

Sau khi được mở rộng biên thì ta có thể tính tích chập trên các phần tử ngoài biên.

Trong hình ảnh ở trên, ta gọi phép tính này là convolution với padding = 1. Padding = k nghĩa là ta thêm k vecto 0 vào mỗi phía của ma trận.

Mục đích của phép tính convolution trên ảnh là làm mờ, làm nét ảnh; xác định các đường;... Mỗi kernel khác nhau thì sẽ phép tính convolution sẽ có ý nghĩa khác nhau.

Bây giờ ta sẽ đi vào một mạng CNN tổng quát.

Giả sử ta có một input image kích thước H*W*D. (phép tính tích chập được tính như nhau trên từng tensor).

*Tầng đầu tiên, như đã đề cập ở trên, đó chính là tầng tính tích chập (Convolutional Layer)

Với kervel có kích thước F*F*D (vì phép tính chập được thực hiện như nhau trên tất cả tensor nên chiều sâu của kernel luôn bằng kernel của input image) và ta thực hiện tính tích chập trên tất cả các phần tử (được gọi là stride, trong trường hợp này thì stride bằng 1).

Mô hình tổng quát: 

<img width="454" alt="NN5" src="https://user-images.githubusercontent.com/51883796/77915719-23ea4900-72c2-11ea-96f1-5204d6bdd114.PNG">

Lưu ý:

- Output của Conv layer này trước khi trở thành input của Conv Layer khác thì sẽ đi qua hàm activation 

- Tổng số parameter (đường dẫn) của 1 layer là K*(F*F*D +1)

*Tầng thứ 2: Pooling Layer 

Mục đích của tầng này: tầng này được đặt ở giữa các Conv layer để làm giảm kích thước dữ liệu nhưng vân giữ được các thuộc tính quan trọng của dữ liệu, từ đó sẽ dễ dàng hơn trong tính toán. 

Thông thường khi dùng pooling layer thì ta sẽ lấy size = (2,2) , stride = 2, padding = 0 để giảm kích thước dữ liệu xuống còn một nữa nhưng vẫn giữ nguyên depth của nó.

<img width="455" alt="NN6" src="https://user-images.githubusercontent.com/51883796/77923359-8f391880-72cc-11ea-883b-312c91e9703a.PNG">

Có 2 loại pooling layer phổ biến: Max pooling hoặc Average pooling

<img width="443" alt="NN7" src="https://user-images.githubusercontent.com/51883796/77923505-ba236c80-72cc-11ea-92bd-daa95d25f401.PNG">

*Tầng tiếp theo: Fully Connected Layer

Sau khi đã đọc được các đặc trung của ảnh thì hì tensor của output của layer cuối cùng, kích thước H*W*D, sẽ được chuyển về 1 vector kích thước (H*W*D)


<img width="433" alt="NN8" src="https://user-images.githubusercontent.com/51883796/77923654-f060ec00-72cc-11ea-928c-25f8cbc892c2.PNG">

Sau đó ta dùng các fully connected layer để kết hợp các đặc điểm của ảnh để ra được output của model.

*Tổng quát:

<img width="443" alt="NN9" src="https://user-images.githubusercontent.com/51883796/77923791-1e463080-72cd-11ea-8143-31776d1c77df.PNG">

Tài liệu và hình ảnh được tham khảo từ: 

https://nttuan8.com/bai-4-backpropagation/

https://machinelearningcoban.com/2017/02/24/mlp/

Nguyễn Thanh Tuấn - "Deep Learning Cơ bản" - Phần III: "Neural Network" , Phần IV: "Convolutional Neural Network" 
