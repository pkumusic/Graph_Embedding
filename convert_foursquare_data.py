__author__ = "Music"
# Read user, venue, category, time? graph
from geopy.distance import great_circle
from collections import defaultdict
import cPickle
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s.%(msecs)d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")

def get_friendship_info(file, userID_index):
    count = 0
    user_user = []
    for line in file:
        if count == 0:
            count += 1
            continue
        user1, user2 = line.strip().split(',')
        user_user.append([userID_index[user1], userID_index[user2]])
        count += 1
    return user_user # index: total number of

def get_venue_info(venue_file):
    userID_index = {}
    index = 0
    count = 0
    venue_file.seek(0)
    for line in venue_file:
        if count == 0:
            count += 1
            continue
        count += 1
        userID, time, venueID, venue_name, venueLocation, venueCategory = line.strip().split('\t')
        # print userID, time, venueID, venue_name, venueLocation, venueCategory
        if userID not in userID_index:
            userID_index[userID] = index
            index += 1

    venueID_index = {}
    venueID_venueInfo = {}
    categories_set = set([])
    category_index = {}

    count = 0
    venue_file.seek(0)
    for line in venue_file:
        if count == 0:
            count += 1
            continue
        count += 1
        userID, time, venueID, venue_name, venueLocation, venueCategory = line.strip().split('\t')
        #print userID, time, venueID, venue_name, venueLocation, venueCategory
        location = venueLocation[1:-1].split(',')
        location = ','.join(location[2:])
        categories = venueCategory[1:-1].strip().split(',')
        if venueID not in venueID_index:
            venueID_index[venueID] = index
            venueID_venueInfo[venueID] = [venue_name[1:-1],location,'|'.join(categories)]
            index += 1
        #print categories
        for category in categories:
            if category != '':
                categories_set.add(category)

    for category in categories_set:
        category_index[category] = index
        index += 1

    # @hzt
    location_index = {}

    count = 0
    venue_file.seek(0)
    for line in venue_file:
        if count == 0:
            count += 1
            continue
        count += 1
        userID, time, venueID, venue_name, venueLocation, venueCategory = line.strip().split('\t')
        location = venueLocation[1:-1].split(',')
        location = ','.join(location[2:])
        if location not in location_index:
            location_index[location] = index
            index += 1

    user_venue = []
    venue_category = set([])
    venue_location = set([])

    count = 0
    venue_file.seek(0)
    for line in venue_file:
        if count == 0:
            count += 1
            continue
        count += 1
        userID, time, venueID, venue_name, venueLocation, venueCategory = line.strip().split('\t')
        #print venueID_index[venueID]
        #print userID_index[userID]
        user_venue.append([userID_index[userID], venueID_index[venueID]])
        #
        categories = venueCategory[1:-1].strip().split(',')
        for category in categories:
            if category != '':
                venue_category.add((venueID_index[venueID], category_index[category]))
        # @hzt
        location = venueLocation[1:-1].split(',')
        location = ','.join(location[2:])
        venue_location.add((venueID_index[venueID], location_index[location]))
    venue_category = list(venue_category)
    venue_location = list(venue_location)

    print('#user %d' % len(userID_index))
    print('#venue %d %d' % (len(venueID_index), len(venueID_venueInfo)))
    print('#category %d %d' % (len(category_index), len(categories_set)))
    print('#location %d' % len(location_index))
    print('#user-venue %i' % len(user_venue))
    print('#venue_category %i' % len(venue_category))
    print('#venue_location %i' % len(venue_location))

    return userID_index, venueID_index, venueID_venueInfo, category_index, \
           location_index, user_venue, venue_category, venue_location


def write_nodetypes_file(dir, userID_index, venueID_index, venueID_venueInfo, category_index, location_index):
    f = open(dir+'node_types.txt', 'w')
    f.write('type 0\n')
    for val in userID_index.values():
        f.write(str(val)+'\n')
    f.write('type 1\n')
    for val in venueID_index.values():
        f.write(str(val) + '\n')
    f.write('type 2\n')
    for val in category_index.values():
        f.write(str(val) + '\n')
    f.write('type 3\n')
    for val in location_index.values():
        f.write(str(val) + '\n')
    f.flush()
    f.close()

    f_user = open(dir+'userID_index.txt', 'w')
    for k,v in userID_index.iteritems():
        f_user.write(k+','+str(v)+'\n')
    f_user.flush()
    f_user.close()

    f_venue = open(dir+'venueID_index.txt', 'w')
    for k,v in venueID_index.iteritems():
        f_venue.write(k+','+str(v)+'\n')
    f_venue.flush()
    f_venue.close()

    f_venueinfo = open(dir+'venue_info.txt', 'w')
    for k,v in venueID_venueInfo.iteritems():
        f_venueinfo.write(k+','+','.join(str(e) for e in v)+','+str(venueID_index[k])+'\n')
    f_venueinfo.flush()
    f_venueinfo.close()

    f_cate = open(dir+'category_index.txt', 'w')
    for k,v in category_index.iteritems():
        f_cate.write(k+','+str(v)+'\n')
    f_cate.flush()
    f_cate.close()

    f_loc = open(dir+'location_index.txt', 'w')
    for k,v in location_index.iteritems():
        f_loc.write(k+','+str(v)+'\n')
    f_loc.flush()
    f_loc.close()


def write_edges_file(dir, user_user, user_venue, venue_category, venue_location):
    f = open(dir + 'edges.txt', 'w')
    f.write('type 0 0\n')
    for tuple in user_user:
        f.write(' '.join(map(str, tuple)) + '\n')
    f.write('type 0 1\n')
    for tuple in user_venue:
        f.write(' '.join(map(str, tuple)) + '\n')
    f.write('type 1 2\n')
    for tuple in venue_category:
        f.write(' '.join(map(str, tuple)) + '\n')
    f.write('type 1 3\n')
    for tuple in venue_location:
        f.write(' '.join(map(str, tuple)) + '\n')
    f.flush()
    f.close()

def cal_geo_distance(point1, point2):
    """ point:  (longitute, latitute) e.g., (40.75594383981997,-73.99811064164594)
        Calculate the distance between two geometric points.
        We only consider two points within 100km to evaluate in the paper
    """
    return great_circle(point1, point2).km

def get_venueID_point(venue_file):
    # return {venueID:point, venueID:point...}
    # city_venueID {city: [venueID, venueID...]}
    # point:  (longitute, latitute)
    venueID_point = {}
    city_vanueID = defaultdict(list)
    venueID_city = {}
    count = 0
    venue_file.seek(0)
    for line in venue_file:
        if count == 0:
            count += 1
            continue
        count += 1
        userID, time, venueID, venue_name, venueLocation, venueCategory = line.strip().split('\t')
        point = (venueLocation[1:-1].split(',')[0], venueLocation[1:-1].split(',')[1])
        city_vanueID[venueLocation[1:-1].split(',')[2]].append(venueID)
        venueID_city[venueID] = venueLocation[1:-1].split(',')[2]
        if venueID not in venueID_point:
            venueID_point[venueID] = point
        else:
            #print "Repeated venue"
            assert venueID_point[venueID] == point # Passed
    return venueID_point, city_vanueID, venueID_city

def locationsInsideCircle(venueID_point, city_vanueID, venueID_city, circle_distance = 100): #100km
    # for each venueID, find all the other venueID that are within the circle of distance
    # return {venueID:[venueID,venueID,...], venueID:[...]}


    location_circle = defaultdict(list)
    keys = venueID_point.keys()

    # This should be the right way. But time complexity is too large
    # count = 0
    # print len(keys), "venues at total"
    # for key1 in keys[:3]:
    #     if count % 10 == 0:
    #         print count, "venue calculated"
    #     count += 1
    #     #print len(city_vanueID[venueID_city[key1]]), "number of venue needs to be calculated"
    #     for key2 in city_vanueID[venueID_city[key1]]:
    #         if key1 == key2: continue
    #         point1, point2 = venueID_point[key1], venueID_point[key2]
    #         dist = cal_geo_distance(point1, point2)
    #         #print dist
    #         if dist < circle_distance:
    #             location_circle[key1].append(key2)

    # The venues in the same city are actually all within 100km......
    for key in keys:
        location_circle[key] = city_vanueID[venueID_city[key]]

    print "Done with building location_circle dict"
    return location_circle


def write_venueID_point_file(dir, venueID_point):
    f_venue = open(dir + 'venueID_point.txt', 'w')
    f_latitude = open(dir + 'latitude.txt', 'w' )
    f_longitude = open(dir + 'longitude.txt', 'w')
    count = 0
    for k, v in venueID_point.iteritems():
        f_latitude.write(v[0] + '\n')
        f_longitude.write(v[1] + '\n')
        f_venue.write(k+'\t'+str(v)+'\n')
        count += 1
    print count
    f_venue.flush()
    f_venue.close()


if __name__ == "__main__":
    venue_file = open('data/foursquare/checkin_CA_venues.txt', 'r')
    #case = 'geo_info'
    case = 'create_file'
    if case == 'create_file':
        userID_index, venueID_index, venueID_venueInfo, category_index, \
        location_index, user_venue, venue_category, venue_location = get_venue_info(venue_file)

        friendship_file = open('data/foursquare/fs_friendship_CA.txt', 'r')
        user_user = get_friendship_info(friendship_file, userID_index)
        ## user_user is bi-direct. extend user_user graph by the next line if needed
        #user_user = user_user + map(lambda x: x[::-1], user_user)
        print('#user-user %i' % len(user_user))

        write_nodetypes_file('data/foursquare/', userID_index, venueID_index, venueID_venueInfo, category_index, location_index)
        write_edges_file('data/foursquare/', user_user, user_venue, venue_category, venue_location)

        print 'Done.'
    if case == 'geo_info':
        # logging.info("start")
        # count = 0
        # for i in xrange(10000*10000): # 1min for 10000 * 10000
        #     count+=1
        #     if count % 10000 == 0: b=1+1
        # logging.info("finished")
        # exit()
        #print cal_geo_distance(('40.75594383981997','-73.99811064164594'), ('40.72262464797742','-73.99995849221705'))
        #print cal_geo_distance(('', ''), ('', ''))
        venueID_point, city_venueID, venueID_city = get_venueID_point(venue_file)
        # write venueID_point file for C to process
        #write_venueID_point_file('data/foursquare/', venueID_point)
        for key in city_venueID:
            if len(city_venueID[key])>10000:
                print key, len(city_venueID[key])
        location_circle = locationsInsideCircle(venueID_point, city_venueID, venueID_city)
        for key in location_circle:
            print len(location_circle[key])
            print len(city_venueID[venueID_city[key]])
        cPickle.dump(location_circle, open("data/foursquare/location_circle.p" , "wb"))
