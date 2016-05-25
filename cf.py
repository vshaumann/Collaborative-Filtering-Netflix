import argparse
import numpy as np
import csv


###### STATS FUNCTIONS ######

def get_intsect(i, j, usr_d):
    '''
    Returns intersection of movies rated by users i and j.
    '''
    i_mov = usr_d[i].keys()
    j_mov = usr_d[j].keys()
    return np.intersect1d(i_mov, j_mov)


def get_usr_rating(usr, int_list, mov_d):
    '''
    Returns a list of a user ratings for movies in intersection list.
    '''
    user_ratings = list()
    for i in int_list:
        user_ratings.append(mov_d[i][usr])
    return user_ratings


def get_r_bar(usr, int_list, mov_d):
    '''
    movie_list: intersection of i and j
    Computes average rating for movies rated by a user.
    '''
    ratings = get_usr_rating(usr, int_list, mov_d)
    return np.sum(ratings)/ float(len(ratings))


def get_w(i, j, usr_d, mov_d):
    '''
    Returns Pearson coefficient as a measure of similarity
    between users i and j.
    '''
    int_list = get_intsect(i, j, usr_d)

    r_i = get_usr_rating(i, int_list, mov_d)
    r_bar_i = get_r_bar(i, int_list, mov_d)

    r_j = get_usr_rating(j, int_list, mov_d)
    r_bar_j = get_r_bar(j, int_list, mov_d)

    top = np.sum((r_i - r_bar_i) * (r_j - r_bar_j))
    bottom = np.sqrt(np.sum((r_i - r_bar_i) ** 2) * np.sum((r_i - r_bar_i) ** 2))

    if top <= 0 or bottom <= 0:
        return 0
    else:
        return top / float(bottom)


def get_user_avg_mov_rat(user, usr_mov_d):
    '''
    Returns average movie ratings for a specific user.
    '''
    ratings = usr_mov_d[user].values()
    return np.sum(ratings) / len(ratings)


def get_avg_mov_rat(mov, mov_usr_d):
    '''
    Returns average movie ratings for a specific movie.
    '''
    ratings = mov_usr_d[mov].values()
    return np.sum(ratings) / len(ratings)


def get_all_user_avg(d):
    '''
    Returns average of second level nested values in the dictionary.
    '''
    sum_list = list()
    keys = d.values()
    for i in keys:
        sum_list.append(i.values())
    sum_list = [item for sublist in sum_list for item in sublist]
    return np.sum(sum_list) / float(len(sum_list))


def predict_ik(i, mov_test, usr_mov, mov_usr, all_usr_avg):
    top = []
    all_w_ij = []
    ex_usr = usr_mov.keys()
    ex_mov = mov_usr.keys()

    j_users_d = mov_usr[mov_test] # return all user:rating who watched the movie

    for j_user in j_users_d: # iterate through keys
        r_jk = np.array(j_users_d[j_user]) # rating for movie k of user j
        r_bar_j = get_user_avg_mov_rat(j_user, usr_mov) # average rating of user j

        w_ij = get_w(i, j_user, usr_mov, mov_usr)
        all_w_ij.append(np.absolute(w_ij))

        top_sum = w_ij * (r_jk - r_bar_j)
        top.append(top_sum)


    # Check for the Cold Start Problem Cases

    # New Movie and New User
    if i not in ex_usr and mov_test not in ex_mov:
        # Set to average rating of all existing users
        r_bar_ik = all_usr_avg
    # New User
    elif i not in ex_usr:
        # Set to the average of the movie
        r_bar_ik = get_avg_mov_rat(mov_test, mov_usr)
    elif mov_test not in ex_mov:
        # Set to the
        r_bak_ik = get_user_avg_mov_rat(i, usr_mov)
    else:
        r_bar_i = get_user_avg_mov_rat(i, usr_mov)

        if np.sum(all_w_ij) == 0:
            all_w_ij = 1
        r_bar_ik = r_bar_i + (np.sum(top) / np.sum(all_w_ij))

    # Check for any NA or abnormal values
    # Set them to the all user average, 1, or 5 where appropriate

    if np.isnan(r_bar_ik) == True:
        r_bar_ik = all_usr_avg

    if abs(r_bar_ik) < 1.:
        r_bar_ik = 1

    if abs(r_bar_ik) > 5.:
        r_bar_ik = 5

    return round(r_bar_ik, 2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, nargs=1)
    parser.add_argument('--test', required=True, nargs=1)
    args = parser.parse_args()

    TRAIN_FILE = args.train[0]
    TEST_FILE = args.test[0]
    WRITE_FILE = "predictions.txt"

    ###### READ DATA ######

    usr_mov = dict()
    mov_usr = dict()

    print "=-" * 20
    print "Reading :", TRAIN_FILE
    with open(TRAIN_FILE , "r") as fh:
        for i in fh:
            mov, usr, rat = i.split(",")
            mov = int(mov)
            usr = int(usr)
            rat = float(rat.strip())

            try:
                usr_mov[usr][mov] = rat
            except KeyError:
                usr_mov[usr] = {mov: rat}

            try:
                mov_usr[mov][usr] = rat
            except KeyError:
                mov_usr[mov] = {usr: rat}

    print "=-" * 30
    print "Done..."

    ##### PREDICTIONS AND TESTING ######

    errors1 = list()
    errors2 = list()

    print "=-" * 30
    print "Making Prediction on: ", TEST_FILE
    print "Number of lines in %s: " % TEST_FILE,  sum(1 for line in open(TEST_FILE))

    all_usr_avg = get_all_user_avg(mov_usr)

    with open(TEST_FILE, "r") as fh:
        for i in fh:
            mov_test, user, actual_rat = i.split(",")
            mov_test = int(mov_test)
            user = int(user)
            actual_rat = float(actual_rat)

            pred = predict_ik(user, mov_test, usr_mov, mov_usr, all_usr_avg)

            error1 = abs(actual_rat - pred)
            errors1.append(error1)

            error2 = (actual_rat - pred) ** 2
            errors2.append(error2)

            with open(WRITE_FILE, "a") as wf:
                writer = csv.writer(wf)
                writer.writerow((mov_test, user, pred))

    mean_abs_error = np.sum(errors1) / len(errors1)
    rmse = np.sqrt(np.sum(errors2) / len(errors2))

    print "=-" * 30
    print "Mean Absolute Error: ", mean_abs_error.round(2)
    print "Root Mean Squared Error: ", rmse.round(2)
