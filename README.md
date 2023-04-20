# Assignment_Data-Science_Hacktiv8_Krisna-Winar-Setiawan
# PYTN_Assgn_1_04_Krisna Winar Setiawan
##Dataset Overview

This data, represented by the file london_crime_by_lsoa.csv, covers the number of criminal reports by month, LSOA borough, and major/minor category from Jan 2008-Dec 2016 in Greater London (central London and the surrounding metropolitan area) by providing 13,490,604 samples with 7 variables each.
The variables lsoa_code, borough, major_category, minor_category, year and month are categorical variables, while value is a discrete numerical variable. The variables' meanings are the followings:
lsoa_code: code for Lower Super Output Area in Greater London; borough: common name for London borough; major_category: high level categorization of crime; minor_category: low level categorization of crime within major category; year: year of reported counts, 2008-2016; month: month of reported counts, 1-12; value: monthly reported count of categorical crime in given borough;
##Analisis Keseluruhan

Dataset ini memiliki 7 Atribut/feature terdapat 9 kategori major dan 32 jenis kejahatan kategori minor dari periode tahun 2008 - 2016. variable value merupakan satu satunya variabel numeric pada dataset. var value mewakili jumlah laporan bulanan kategori kejahatan pada setiap wilayah dengan unique valuenya 247, maximal value 309 dan minimal value adalah 0 . Data hasil Pie chart diatas kita dapat melihat bahwa kejahatan dengan jumlah tertinggi pada major crime category adalah Theft and Handling dengan persentase 41.3%, dan diikuti Violence Against the Person yang mengambil tempat kedua dengan persentase 24.2%. Pada Bar chart diatas menampilkan hasil bahwa Theft from motor Vehicle, common assault dan pencurian lainnya menjadi kejahatan kategori minor terbanyak dengan total kasus 522180 dan jumlah kejahatan kategori minor paling sedikit kasusnya adalah Rape dengan total kasus 27000 per tahun 2008 - 2016

# PYTN_Assgn_2_04_Krisna Winar Setiawan
## Data Overview

Dataset ini bernama nyc-rolling-sales.csv yang berisi lokasi, alamat, tipe, harga jual, dan tanggal penjualan unit bangunan yang terjual. Referensi di bidang yang lebih rumit:

BOROUGH : Kode digit untuk borough tempat properti berada; agar ini adalah Manhattan (1), Bronx (2), Brooklyn (3), Queens (4), dan Staten Island (5).
MEMBLOKIR; BANYAK : Kombinasi borough, blok, dan lot membentuk kunci unik untuk properti di New York City. Biasa disebut BBL.
KELAS BANGUNAN SAAT INI dan KELAS BANGUNAN SAAT DIJUAL : : Jenis bangunan pada berbagai titik waktu.
Perhatikan bahwa karena ini adalah kumpulan data transaksi keuangan, ada beberapa hal yang perlu diingat:

Banyak penjualan terjadi dengan jumlah dolar yang sangat kecil: $0 paling sering. Penjualan ini sebenarnya adalah pengalihan akta antar pihak: misalnya, orang tua mengalihkan kepemilikan rumah mereka kepada seorang anak setelah pindah untuk pensiun.
Kumpulan data ini menggunakan definisi keuangan dari sebuah bangunan/unit bangunan, untuk tujuan perpajakan. Dalam hal satu entitas memiliki bangunan tersebut, penjualan mencakup nilai seluruh bangunan. Jika sebuah bangunan dimiliki sedikit demi sedikit oleh penghuninya (kondominium), penjualan mengacu pada satu apartemen (atau sekelompok apartemen) yang dimiliki oleh beberapa individu.

##Analisis Keseluruhan
Pada Assigment ke 2 ini sya melakukan implementasi Mean,Median,Modus,Range,Variance, Spread Standard, Menganalisa distribusi, Menarik kesimpulan, Melakukan preproces dataset agar dapat digunakan dari Kolom/Data spesifik menggunakan pandas,numpy,scipy
