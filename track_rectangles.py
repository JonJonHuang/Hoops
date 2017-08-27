import csv
import math

class Rectangle:

    def __init__(self, x1, y1, x2, y2, id):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.updated = False
        self.id = id

# define distance as distance between rectangle centers
# midpt1 = ((x2 + x1) / 2, (y2 + y2) / 2)
# distance = dist(midpt1, midpt2)
def calculate_rectangle_distance(rect1, rect2):
    midpt1 = ((rect1.x1 + rect1.x2) / 2, (rect1.y1 + rect1.y2) / 2)
    midpt2 = ((rect2.x1 + rect2.x2) / 2, (rect2.y1 + rect2.y2) / 2)

    distance = math.pow(math.pow((midpt1[0] - midpt2[0]),2) + math.pow(midpt1[1] - midpt2[1], 2), .5)

    return distance

if __name__ == "__main__":

    rectangles = []

    with open('players.csv', 'rb') as csvFile:
        csvReader = csv.reader(csvFile)

        current_rectangles = []
        prev_frame = -1
        i = 0
        for row in csvReader:
            i += 1
            if prev_frame != int(row[0]):
                rectangles.append(current_rectangles)
                current_rectangles = []
                prev_frame = int(row[0])
            current_rectangles.append(Rectangle(int(row[1]), int(row[2]), int(row[3]), int(row[4]), i))

    rectangle_map = {}

    # deque of max size 5
    past_rects = []
    cur_rects = []
    threshold = 50.0

    for i in range(0, len(rectangles)):
        cur_rects = rectangles[i]

        for rect1 in cur_rects:
            min_distance = 1000000
            min_rect = None
            for j in range(len(past_rects) - 1, -1, -1):
                match_found = False
                for rect2 in past_rects[j]:
                    distance = calculate_rectangle_distance(rect1, rect2)
                    if distance < min_distance:
                        min_distance = distance
                        min_rect = rect2
                
                if min_distance < threshold * (len(past_rects) - j + 1):
                    rect1.id = rect2.id
                    match_found = True

                if match_found:
                    j = -1

            if not rectangle_map.has_key(rect1.id):
                rectangle_map[rect1.id] = [rect1]
            else:
                rectangle_map[rect1.id].append(rect1)
        # add the most recent rectangles and delete the oldest one if neccesary
        past_rects.append(cur_rects)
        if len(past_rects) > 5:
            del past_rects[0]

print('hi')
