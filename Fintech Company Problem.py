   # Mengimpor library yang diperlukan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Mengimport dataset
mydata = pd.read_csv('data_fintech.csv')

# Ringkasan data
ringkasan = mydata.describe()
tipe_data = mydata.dtypes

# Merevisi kolom num_screens
mydata['screen_list'] = mydata.screen_list.astype (str) + ','
mydata['num_screens'] = mydata.screen_list.str.count (',')
mydata.drop (columns =['numscreens'], inplace=True)

# Cek kolom hour
mydata.hour[1]
mydata.hour = mydata.hour.str.slice (1,3).astype(int)

# Mendefinisikan variable khusus numerik 
mydata_numerik = mydata.drop(columns=['user','first_open','screen_list',
                                       'enrolled_date'], inplace=False)

# Membuat histrogram
sns.set()
plt.suptitle('Histogram Data Numerik')
for i in range(0, mydata_numerik.shape[1]):
    plt.subplot(3,3,i+1)
    figure = plt.gca()
    figure.set_title(mydata_numerik.columns.values[i])
    jumlah_bin = np.size(mydata_numerik.iloc[:,i].unique())
    plt.hist(mydata_numerik.iloc[:,i], bins=jumlah_bin)
    
# Membuat Correlation Matrix
correlation = mydata_numerik.drop (columns=['enrolled'], inplace=False).corrwith(mydata_numerik.enrolled)
correlation.plot.bar(title='Korelasi Variable terhadap keputusan enrolled')

matrix_correlation = mydata_numerik.drop(columns=['enrolled'], inplace=False).corr()
sns.heatmap(matrix_correlation, cmap = 'Blues')

mask = np.zeros_like(matrix_correlation, dtype = np.bool)
mask [np.triu_indices_from(mask)] =True #triu for upper, tril for lower

# Membuat correlation matrix dengan heatmap custom
ax = plt.axes() #canvas kosong
cmap_ku = sns.diverging_palette(200,0,as_cmap=True)
sns.heatmap(matrix_correlation,cmap=cmap_ku, mask=mask,
            linewidths= 0.5, center = 0, square = True)
ax = plt.suptitle('Corelation Matrix Custom')

# Feature Engineering (menyeleksi fitur)
# Prosses Parsing
from dateutil import parser
mydata.first_open = [parser.parse(i) for i in mydata.first_open] 
mydata.enrolled_date = [parser.parse(i) if isinstance (i, str) else i for i in mydata.enrolled_date]
mydata ['selisih'] = (mydata.enrolled_date - mydata.first_open).astype('timedelta64[h]')

# Membuat histogram selisih my data
plt.hist(mydata.selisih.dropna(), range = [0,200])
plt.suptitle('selisih waktu antara enrolled dengan first open')
plt.show()

# Memfilter nilai selisih >48 hours
mydata.loc[mydata.selisih>48, 'enrolled'] = 0

# Mengimpor top screens
top_screens = pd.read_csv('top_screens.csv')
top_screens = np.array(top_screens.loc[:,'top_screens'])

# Membuat cadanga data
mydata2 = mydata.copy()

# Membuat kolom untuk setiap top_screen
for layar in top_screens :
    mydata2 [layar] = mydata2.screen_list.str.contains(layar).astype(int)
    
for layar in top_screens :
    mydata2 ['screen_list'] = mydata2.screen_list.str.replace(layar+',','')
    
# Menghitung item non top screens di screen_list
mydata2['lainnya'] = mydata2.screen_list.str.count(',')

# Proses penggabungan beberapa screen yang sama  (funneling)
layar_loan = ['Loan',
              'Loan2',
              'Loan3',
              'Loan4']
mydata2 ['jumlah_loan'] = mydata2[layar_loan].sum(axis=1)
mydata2.drop(columns = layar_loan, inplace = True)

layar_saving = ['Saving1',
                'Saving2',
                'Saving2Amount',
                'Saving4',
                'Saving5',
                'Saving6',
                'Saving7',
                'Saving8',
                'Saving9',
                'Saving10']
mydata2 ['jumlah_saving'] = mydata2[layar_saving].sum(axis=1)
mydata2.drop(columns = layar_saving, inplace = True)

layar_credit = ['Credit1',
                'Credit2',
                'Credit3',
                'Credit3Container',
                'Credit3Dashboard']
mydata2 ['jumlah_credit'] = mydata2[layar_credit].sum(axis=1)
mydata2.drop(columns = layar_credit, inplace = True)

layar_cc = ['CC1',
            'CC1Category',
            'CC3']
mydata2 ['jumlah_CC'] = mydata2[layar_cc].sum(axis=1)
mydata2.drop(columns = layar_cc, inplace = True)

# Mendefinisikan var dependen
var_enrolled = np.array(mydata2['enrolled'])

# Menghilangkan columns yang redundan
mydata2.drop(columns = ['user', 'first_open', 'screen_list','enrolled',
                        'enrolled_date', 'selisih'], inplace = True)

# Mmebagi menjadi training dan test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(np.array (mydata2), var_enrolled,
                                                    test_size= 0.2,
                                                    random_state=111)

# Pre Processing Standarisasi (feature scaling)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Menghilangkan variable kosong
X_train = np.delete(X_train, 27,1)
X_test = np.delete(X_test, 27,1)

# logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state= 0, solver = 'liblinear',
                                penalty= 'l1')

classifier.fit(X_train, y_train)

# Memprediksi test set
y_pred = classifier.predict(X_test)

# Mengevaluasi model dengan confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print (classification_report(y_test, y_pred))

# Menggunakan accuracy_score
evaluasi = accuracy_score(y_test, y_pred)
print ('Akurasi:{:.2f}'.format(evaluasi*100))

# Menggunakan seaborn untuk CM
cm_label = pd.DataFrame(cm, columns = np.unique(y_test),
                        index = np.unique(y_test))
cm_label.index.name = 'Aktual'
cm_label.columns.name = 'Prediksi'
sns.heatmap(cm_label, annot=True, cmap='Blues', fmt='g')

# validasi dengan 10-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X=X_train, y=y_train, cv=10)

accuracies.mean()
accuracies.std()
print ('akurasi regresi logistik ={:.2f}% +/- {:.2f}%'.format(accuracies.mean()*100, accuracies.std()*100)) 

# Mmebagi menjadi training dan test set cari user ID
var_enrolled =(mydata2['enrolled'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(mydata2, var_enrolled,
                                                    test_size= 0.2,
                                                    random_state=111)

train_id = X_train['user']
test_id = X_test ['user']

# menggabungkan Semuanya
y_pred_series = pd.Series(y_test).rename ('asli', inplace=True)
hasil_akhir = pd.concat([y_pred_series, test_id], axis=1).dropna()
hasil_akhir['prediksi'] = y_pred
hasil_akhir = hasil_akhir[['user','asli','prediksi']].reset_index(drop=True)

