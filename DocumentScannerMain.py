import cv2
import numpy as np
import utlis

pathImage = "tester1.jpg"
cap = cv2.VideoCapture(1) # Pastikan indeks kamera HP/webcam sudah sesuai
cap.set(10, 160)
heightImg = 640
widthImg = 480

utlis.initializeTrackbars()
count = 0

while True:
    if webCamFeed:
        success, img = cap.read()
        # AMAN: Jika kamera gagal membaca frame, lewati iterasi ini agar tidak error
        if not success or img is None:
            print("Menunggu sinyal kamera...")
            continue
    else:
        img = cv2.imread(pathImage)
        if img is None:
            print(f"Gagal membaca gambar dari path: {pathImage}")
            break

    img = cv2.resize(img, (widthImg, heightImg))  # resize image
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # buat kanvas kosong
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert ke grayscale
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # fungsi gaussian blur
    thres = utlis.valTrackbars()  # GET TRACK BAR VALUES FOR THRESHOLDS
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])  # tambah canny blur
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)  # operasi dilasi
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # operasi erosi

    ## FIND ALL CONTOURS
    imgContours = img.copy()  
    imgBigContour = img.copy()  
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) 

    # Inisialisasi variabel agar aman digunakan saat tombol 's' ditekan
    imgWarpColored = img.copy() 
    documentDetected = False

    # FIND THE BIGGEST CONTOUR
    biggest, maxArea = utlis.biggestContour(contours) 
    if biggest.size != 0:
        documentDetected = True
        biggest = utlis.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20) 
        imgBigContour = utlis.drawRectangle(imgBigContour, biggest, 2)
        pts1 = np.float32(biggest) 
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]]) 
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        # REMOVE 20 PIXELS FROM EACH SIDE
        imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))

        # APPLY ADAPTIVE THRESHOLD
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)

        # Image Array for Display
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre])
    else:
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

    # LABELS FOR DISPLAY
    lables = [["Original", "Gray", "Threshold", "Contours"],
              ["Biggest Contour", "Warp Prespective", "Warp Gray", "Adaptive Threshold"]]

    stackedImage = utlis.stackImages(imageArray, 0.75, lables)
    cv2.imshow("Result", stackedImage)

    # SAVE IMAGE WHEN 's' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        if documentDetected:
            # Pastikan folder 'Scanned' sudah dibuat sebelumnya di direktori proyek Anda
            cv2.imwrite("Scanned/myImage" + str(count) + ".jpg", imgWarpColored)
            cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.imshow('Result', stackedImage)
            cv2.waitKey(300)
            count += 1
            print(f"Berhasil menyimpan: Scanned/myImage{count-1}.jpg")
        else:
            print("Gagal menyimpan: Dokumen tidak terdeteksi oleh kontur!")
            
    # Keluar dari program jika tombol 'q' ditekan
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()