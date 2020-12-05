import cv2

def main():
    img = cv2.imread('input/input006.jpeg')
    # cv2.cvtColor(img, cv2.COLOR_BGR2YUV, img)
    # channels = cv2.split(img)
    # cv2.equalizeHist(channels[0], channels[0])
    # cv2.merge(channels, img)
    # cv2.cvtColor(img, cv2.COLOR_YUV2BGR, img)
    cv2.equalizeHist(img, img)
    img = cv2.resize(img, dsize=(0, 0), fx=0.2, fy=0.2)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
