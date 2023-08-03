# Mini Project 4 - Predict Customer Clicked Ads Classification by Using Machine Learning
  

## Ringkasan 
Sebuah perusahaan ingin mengetahui efektifitas sebuah iklan yang mereka tayangkan, hal ini penting bagi perusahaan yang bergerak di bidang consultant digital marketing agar dapat mengetahui seberapa besar ketercapainnya iklan yang dipasarkan sehingga dapat menarik customers untuk melihat iklan. Pada mini project ini, beberapa tugas yang perlu dilakukan adalah mencari insight terkait perilaku user dari data tersebut dengan membuatkan visualisasinya, membuat sebuah machine learning yang relevan dengan kebutuhan perusahaan, serta membuat rekomendasi dari hasil penemuan-penemuan yang didapat.
#### Background
Pada mini project ini akan dilakukan pemodelan machine learning, untuk memprediksi potential user dalam digital advertising.
#### Problem
Tim bisnis ingin mengoptimalkan metode cara beriklan mereka di platform digital agar mendapat kan user yang potential untuk click sebuah product. Agar cost yang akan dikeluarkan tidak terlalu besar.
#### Goals
Membuat machine learning model yang dapat mendeteksi potential user untuk convert atau tertarik pada sebuah iklan. Sehingga dapat mengoptimalkan cost dalam beriklan di platform digital.

### 1. Customer Type and Behaviour Analysis on Advertisement
#### Univariate Analysis
a) Click on Ad Distribution
![image](https://github.com/hadasadida/Mini-Project-4-Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/124650679/33d23051-da50-4cef-951a-d17fec1246b9)
Target (Click on Ad) memiliki jumlah yang cukup balanced (seimbang) sehingga tidak perlu melakukan preprocessing untuk mengatasi imbalanced class.

b) Age Distribution
![image](https://github.com/hadasadida/Mini-Project-4-Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/124650679/d8a58cf1-420d-4207-95a9-7a76b75162f9)
Berdasarkan plot, umur 30 tahun keatas lebih banyak melakukan click on ad dibandingkan dengan umur 30 tahun kebawah. sehingga dapat disimpulkan bahwa banyak orang tua yang melakukan click on ad dibandingkan anak muda karena anak muda lebih selektif dalam melakukan click on ad.

c) Daily Time Spent on Site Distribution
![image](https://github.com/hadasadida/Mini-Project-4-Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/124650679/cf137ab6-4319-4c43-b2fa-cd74f3efb923)
Berdasarkan plot, user yang mengunjungi website dengan waktu sebentar lebih banyak melakukan click on ad dibandingkan user yang mengunjungi website dengan waktu yang lama.

d) Daily Internet Usage on Site Distribution
![image](https://github.com/hadasadida/Mini-Project-4-Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/124650679/d543f27d-6bd7-407b-9bab-7f6647d4b1e0)
Berdasarkan plot, user yang jarang menggunakan internet lebih banyak melakukan click on ad dibandingkan pengguna yang sering menggunakan internet. hal ini dapat memberi pentunjuk bahwa user yang jarang menggunakan internet lebih memperhatikan ads.
daily internet usage dan daily time spent on site memiliki distribusi yang mirip.

e) Daily Internet Usage vs Daily Time Spent on Site
![image](https://github.com/hadasadida/Mini-Project-4-Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/124650679/3c2d9089-e069-4270-aff2-566a9fbfa816)
Berdasarkan plot di atas, terdapat 2 kelompok yaitu user aktif internet dan user non aktif internet. user aktif cenderung tidak melakukan click on ad sedangkan user non aktif cenderung melakukan click on ad.
sehingga dapat dilakukan pengoptimalan sistem advertisment terhadap user yang tidak aktif menggunakan internet.

f) Age vs Daily Time Spent on Site
![image](https://github.com/hadasadida/Mini-Project-4-Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/124650679/9da9463d-684e-4b26-9910-aacb749c6c01)
Berdasarkan plot di atas, terdapat 2 kelompok yaitu pertama user dengan umur < 40 tahun dengan mengunjungi website lebih lama dan kedua user dengan umur > 40 tahun dengan mengunjungi website sebentar. kelompok kedua dengan user umur > 40 tahun dan mengunjungi website lebih lama cenderung melakukan click on ad.

g) Age vs Daily Internet Usage
![image](https://github.com/hadasadida/Mini-Project-4-Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/124650679/fea04870-64ce-4da8-9bde-74df9ff7a8c2)
Age vs daily time spent on site dan age vs daily internet usage memiliki 2 kelompok yang mirip yaitu user dengan user > 40 tahun dan jarang menggunakan internet lebih banyak melakukan click on ad.
sehingga dapat dilakukan pengoptimalan sistem advertisment pada user-user yang berumur lebih dari 40 tahun.

#### Multivariate Analysis
![image](https://github.com/hadasadida/Mini-Project-4-Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/124650679/eda9f920-6160-4c4d-a0bb-058663f71d48)
Dari korelasi di atas tidak menemukan adanya multicorrelation (korelasi antar variable) sehingga dapat menggunakan semua feature untuk dilakukan modeling. Namun dengan menggunakan korelasi pearson tidak dapat mengetahui hubungan antara feature dengan targetnya. Maka perlu menggunakan PPS (Predictive Power Score) dalam menghitung hubungan antar feature dengan targetnya.

##### Correlation using PPScore
![image](https://github.com/hadasadida/Mini-Project-4-Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/124650679/808d5026-f3e6-4168-b245-251e79a01ff9)
Feature yang cukup berhubungan dengan target adalah
1) Age
2) Area income
3) daily internet usage
4) daily time spent on site

### 2. Data Cleaning & Preprocessing
#### Data Duplicated
Tidak ada data duplikat pada dataset.
#### Data Null
Terdapat data null pada kolom Daily Time Spent on Site, Area Income, Daily Internet Usage dan Male. Karena jumlah data null sedikit maka dilakukan penghapusan.
#### Feature Encoding
Dilakukan feature encoding pada kolom dengan data type object.
#### Extract Column with Data Type datetime
Membuat kolom baru untuk mengekstraksi data waktu menjadi tahun, bulan, pekan, dan hari.
#### Split Data
Split data menjadi feature dan target.

### 3. Data Modeling
#### Split Data Train and Data Test
Membagi data secara terpisah menjadi data train 70% dan data test 30%.
#### Modeling
Membuat dua eksperimen model machine learning dengan beberapa jenis algoritma model machine learning, dimana eksperimen pertama tanpa menggunakan normalisasi data, dan yang kedua menggunakan normalisasi data.

a) Tanpa Normalisasi Data
![image](https://github.com/hadasadida/Mini-Project-4-Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/124650679/ebae3904-2030-4e5c-b6a2-64b7a0429aaf)
Berdasarkan hasil di atas, model random forest dan gradient boosting memiliki nilai akurasi paling tinggi daripada model yang lain. 
Sedangkan model logistic regession dan k-nearest neighbor memiliki nilai akurasi yang tidak begitu bagus.

b) Dengan Normalisasi Data
![image](https://github.com/hadasadida/Mini-Project-4-Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/124650679/01464821-6dd6-41f5-b504-00b2f7f767a9)
Berdasarkan hasil di atas, dengan melakukan normalisasi data, model logistic regression mengalami kenaikan nilai akurasi yang cukup signifikan. Model logistic regression, random forest dan gradient boosting memiliki nilai akurasi paling tinggi dibandingkan model yang lain. sehingga dapat dipilih model random forest sebagai model terbaik karena memiliki nilai paling bagus dan model logistic regression dapat menjadi pilihan kedua apabila terdapat kendala dalam komputasi.

#### 3) Evaluation
#### Confusion Matrix
![image](https://github.com/hadasadida/Mini-Project-4-Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/124650679/2ab711ab-aeb4-4010-bab7-31ae13038e58)

Confusion matrix yang dihasilkan dari model random forest sangat baik. Kesalahan prediksi (pada cell ungu) memiliki jumlah sangat sedikit (bagian kanan atas dan kiri bawah). Dengan hasil berikut maka akan mendapatkan akurasi, precision, dan recall yang bagus.

#### Feature Importance 
![image](https://github.com/hadasadida/Mini-Project-4-Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/124650679/8c24bc6c-d29c-451f-8543-8155d75b19dd)

Berdasarkan metode random forest dapat dilihat bahwa daily internet usage merupakan feature yang sangat penting dalam menentukan apakah user akan click atau tidak. Sedangkan feature penting lain adalah daily time spent on site, income suatu area dan umur.

Jika dikombinasikan dengan insight yang diperoleh dari proses EDA, ternyata penggunaan internet harian jika semakin tinggi maka peluang user akan click semakin kecil.

### 4. Business Recomendation
#### Kesimpulan
1. Terdapat 2 segment user yaitu kelas satu dan kelas dua, dimana
   - Kelas satu memiliki kebiasaan sering menggunakan internet, sering mengunjungi website suatu product, pendapatan yang tinggi dan 
     umurnya yang relatif muda.
   - Sedangkan kelas 2 memiliki kebiasaan sebaliknya.
2. User yang sangat sering menggunakan internet lebih sulit untuk diberikan iklan karena mereka mungkin sudah terbiasa terhadap digital 
   ads sehingga lebih selektif dalam mengklik suatu ads.
3. User dengan kelompok orang tua merupakan market yang potensial untuk menjadi market digital.
#### Action
1. Menggunakan konsep advertisment yang berbeda agar tidak terlihat seperti ads pada umumnya sehingga user dengan segment kelas satu bisa tertarik melakukan click on ads.
2. Menggunakan konsep konten yang sederhana namun dapat menjadi topik pembicaraan agar dapat memikat user.


