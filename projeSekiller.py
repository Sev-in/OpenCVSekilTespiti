import cv2
import numpy as np

#ŞEKİL TESPİTİ - bu yüzden renklerin bir önemi yok

#1)imread = resmi piksel piksel okuduk.
#2)cvtColor = pikselleri gri tonlarına çevirdik.
#3)Canny = piksellerin renklerinin net bir şekilde ayrıldığı yerler yani kenarlar belirlendi.
#4)findContours = her bir kontürün koordinatları alındı ve listeye eklendi.
#5)contourArea = kontürlerin kapladıkları alanlar hesaplanarak gürültülü verilerden arınıldı.
#6)approxPolyDP = kontürlerin köşelerini bulduk.
#7)boundingRect = kontürlerin etrafına görünmez bir dikdörtgen çizdik ki yazı konumumuz belli olsun.
#8)drawCountours = resmin etrafındaki kontürleri renklendirdik.

img = cv2.imread("OpenCV/proje01/sekiller.png")
cv2.imshow("Başlangıç",img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray,70,150) #150 = üst eşik değeri, 80 = alt eşik değeri canny= kurnaz,açıkgöz,güzel
contours , _= cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

for cont in contours:
    area = cv2.contourArea(cont) # cv2.contourArea(): Verilen bir konturun kapladığı alanı hesaplar. bu area sayesinde çok küçük gereksiz yani gürültü olan nesneleri resimden çıkarmış olduk.
    if area>1000:
        approx = cv2.approxPolyDP(cont, 0.01 * cv2.arcLength(cont,True),True) # köşelerini buluyor
        cornerCount = len(approx) # köşe sayısı
        x,y,w,h = cv2.boundingRect(approx) # yapıyı görünmez bir dikdörtgen içerisine alıyor. yazı konumu için
        if(cornerCount==3):
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1) #etrafına çizilen dikdörtgeni görünür yapmış olduk
            cv2.putText(img,"Ucgen",(x+10,y+10),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
        elif(cornerCount==4):
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
            cv2.putText(img,"Dortgen",(x+10,y+10),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
            cv2.putText(img,"Bilinmeyen",(x+10,y+10),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
        cv2.drawContours(img,cont,-1,(0,255,0),3) #-1 =  Bu değer -1 ise konturun tüm parçalarını çizer. 
    # bu sayede img deki her bir nesnenin sınırları yeşile boyandı
    

cv2.imshow("Sonuç",img)
cv2.imshow("Siyah-Beyaz",canny)

cv2.waitKey(0)

# 🎯 Canny algoritması ne yapıyor?
# Görüntüdeki kenarları bulmak için önce her pikselin gradyan şiddetini hesaplar. Bu, bir pikselin çevresine göre ne kadar değiştiğini (ani parlaklık farkı olup olmadığını) ölçer.
# Bu gradyan değeri ne kadar büyükse, orada keskin bir kenar vardır demektir.

# YANİ ÜÇGENİN RENGİ ARKAPLAN RENGİ İLE DAHA YAKIN OLDUĞU İÇİN EŞİK DEĞERİNİ DÜŞÜRMEMİZ GEREKTİ.


# 🔍 Nedir bu findContours?
# cv2.findContours(), siyah-beyaz (binary) bir görüntüdeki beyaz alanların çevresini takip ederek kontur (sınır çizgileri) oluşturur. Bu genelde cv2.Canny() gibi işlemlerden sonra kullanılır.
# contours, _ = cv2.findContours(
#     canny,               # Girdi görüntü (binary olmalı → kenarlar beyaz)
#     cv2.RETR_EXTERNAL,   # Sadece dış konturları al (iç içe nesneler varsa içtekileri almaz) external = harici
#     cv2.CHAIN_APPROX_NONE  # Her piksel noktasını kaydet (kontur çizgisi tüm detayları içerir) chain = zincir
# approximately = yaklaşık olarak
# )
# _: Hiyerarşi bilgisi döner ama burada kullanılmadığı için _ ile geçilmiş.

