# import cv2
# import requests
# img = cv2.imread('test2.jpg')
# files = {'image': open('test2.jpg', 'rb')}
# result = requests.post("http://ec2-18-138-214-93.ap-southeast-1.compute.amazonaws.com:8081/detect_obj", files=files).json()
# print(result)
# for p in result["rois"]:
#     cv2.rectangle(img, (int(p[0]),int(p[1])), (int(p[2]),int(p[3])),(255,0,0),2)
# cv2.imshow("image",img)
# cv2.waitKey(0)

# type_list=set()
# with open("cat2id.txt","r",encoding="utf-8") as lines:
#     for line in lines:
#         type_list.add(line.strip())
# fw = open("name2id.txt","w",encoding="utf-8")
# index = 0
# for w in list(type_list):
#     fw.write(w+","+str(index))
#     fw.write("\n")
#     index = index + 1
# print(len(type_list))
# fw.close()
# rows = []
# with open("all_data.csv","r",encoding="utf-8") as lines:
#     for line in lines:
#         rows.append(line.strip())
# from sklearn.model_selection import train_test_split

# X_train, X_test = train_test_split(rows,test_size=0.1, random_state=42)
# fw = open("train_all.csv","w",encoding="utf-8")
# for r in X_train:
#     fw.write(r)
#     fw.write("\n")
# fw.close()

# fw = open("valid_all.csv","w",encoding="utf-8")
# for r in X_test:
#     fw.write(r)
#     fw.write("\n")
# fw.close()

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    Returns
    -------
    float
        in [0, 1]
    """

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def select_box(boxes,scores,labels,score_thresh=0.5):
    show_lb = []
    show_box = []
    show_scores = []
    for i in range(len(labels)):
        if (labels[i] not in show_lb) and scores[i]>score_thresh:
            show_lb.append(labels[i])
            show_box.append(boxes[i])
            show_scores.append(scores[i])
        else:
            if(scores[i]>score_thresh):
                index = show_lb.index(labels[i])
                if(scores[i]>=show_scores[index]):
                    show_box[index] = boxes[i]
                    show_scores[index] = scores[i]
    show_lb_1 = [show_lb[0]]
    show_box_1 = [show_box[0]]
    show_scores_1 = [show_scores[0]]
    for i in range(1,len(show_box)):
        check = 0
        for j in range(len(show_box_1)):
            if get_iou(show_box[i],show_box_1[j]) > 0.6:
                if(show_scores[i] >= show_scores_1[j]):
                    show_box_1[j] = show_box[i]
                    show_scores_1[j] = show_scores[i]
                    show_lb_1[j] = show_lb[i]
                    check = 1
        if(check==0):
            show_box_1.append(show_box[i])
            show_lb_1.append(show_lb[i])
            show_scores_1.append(show_scores[i])
    return show_box_1,show_scores_1,show_lb_1


print(get_iou([154.42801,94.63901,240.28914,143.39912],[156.46725,95.41969,239.80753,141.38321]))