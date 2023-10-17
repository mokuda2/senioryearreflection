import itertools
import numpy as np
import random

def add_marriage_edges(list people, list prev_people, int num_people, dict marriage_probs,
                       float prob_marry_immigrant, float prob_marry, double[:, ::1] D,
                       dict indices, list original_marriage_dist,
                       double tol=0, double eps=1e-7):
    cdef dict finite_marriage_probs = {key: val for key, val in marriage_probs.items() if key > 0}
    cdef list desired_finite_distances = [distance for distance, prob in finite_marriage_probs.items() if prob > 0]
    cdef int minimum_permissible_distance = min(k for k in original_marriage_dist if k > -1)
    cdef list marriage_distances = []
    cdef set unions = set()

    cdef set people_set = set(people)
    cdef int next_person = num_people + 1
    cdef int num_inf_couples_to_marry = round(prob_marry_immigrant * len(people) / 2)
    cdef int num_finite_couples_to_marry = round(prob_marry * len(people) / 2)
    cdef set will_marry = set(random.sample(people_set, len(people_set) // 2))
    cdef list wont_marry_until_next_time = [node for node in people_set if node not in will_marry]

    people_set |= will_marry | set(prev_people)

    cdef set possible_couples = {(man, woman) for man, woman in itertools.combinations(people_set, 2)} - {(man, woman) for man, woman in itertools.combinations(prev_people, 2)}

    cdef dict possible_finite_couples = {(man, woman): D[indices[<int>man], indices[<int>woman]] for man, woman in possible_couples if D[indices[<int>man], indices[<int>woman]] >= minimum_permissible_distance}

    cdef dict preferred_couples, other_couples

    preferred_couples = {couple: distance for couple, distance in possible_finite_couples.items() if
                         distance in set(original_marriage_dist) and distance in desired_finite_distances}
    other_couples = {couple: distance for couple, distance in possible_finite_couples.items() if
                     couple not in preferred_couples}

    cdef int iter = 0

    while possible_finite_couples and iter < num_finite_couples_to_marry:
        if preferred_couples:
            possible_finite_couples_array = list(preferred_couples.keys())
            dis_probs = [finite_marriage_probs.get(d, eps) for d in preferred_couples.values()]
            dis_probs = [0 if abs(d) < tol else d for d in dis_probs]  # prevent "negative zeros"
            total_prob = sum(dis_probs)
            dis_probs = [d / total_prob for d in dis_probs]  # normalize

            try:
                couple_index = possible_finite_couples_array[np.random.choice(len(preferred_couples), p=dis_probs)]
            except Exception as e:
                print(f'Error: {e}')

            couple = (couple_index[0], couple_index[1])
        else:
            possible_finite_couples_array = list(other_couples.keys())
            least_bad_distance = min(d for d in other_couples.values() if d >= minimum_permissible_distance)
            dis_probs = [1 if d == least_bad_distance else 0 for d in other_couples.values()]
            dis_probs = [0 if abs(d) < tol else d for d in dis_probs]  # prevent "negative zeros"
            total_prob = sum(dis_probs)
            dis_probs = [d / total_prob for d in dis_probs]  # normalize

            couple_index = possible_finite_couples_array[np.random.choice(len(other_couples), p=dis_probs)]
            couple = (couple_index[0], couple_index[1])

        unions.add(couple)
        marriage_distances.append(int(D[indices[<int>couple[0]], indices[<int>couple[1]]]))

        possible_finite_couples = {pair: distance for pair, distance in possible_finite_couples.items() if couple[0] not in pair and couple[1] not in pair}
        preferred_couples = {couple: distance for couple, distance in possible_finite_couples.items() if distance in set(original_marriage_dist) and distance in desired_finite_distances}
        other_couples = {couple: distance for couple, distance in possible_finite_couples.items() if couple not in preferred_couples}

        iter += 1

    if iter == 0:
        stay_single_forever = will_marry | set(prev_people)
    else:
        stay_single_forever = set(man for couple in possible_couples for man in couple)

    cdef dict possible_inf_couples = {(man, woman): D[indices[man], indices[woman]]
                            for man, woman in itertools.combinations(stay_single_forever, 2)
                            if D[indices[man], indices[woman]] == -1}

    iter = 0
    while possible_inf_couples and iter < num_inf_couples_to_marry:
        possible_inf_couples_array = list(possible_inf_couples.keys())
        couple_index = np.random.choice(len(possible_inf_couples))  # draw uniformly
        couple = (possible_inf_couples_array[couple_index][0], possible_inf_couples_array[couple_index][1])

        unions.add(couple)
        marriage_distances.append(int(D[indices[<int>couple[0]], indices[<int>couple[1]]]))

        possible_inf_couples = {pair: distance for pair, distance in possible_inf_couples.items() if couple[0] not in pair and couple[1] not in pair}

        iter += 1
        stay_single_forever = set(man for couple in possible_couples for man in couple)

    num_immigrants = num_inf_couples_to_marry - iter
    num_immigrants = min(len(stay_single_forever), num_immigrants)

    immigrants = list(range(next_person, next_person + num_immigrants))
    marry_strangers = np.random.choice(list(stay_single_forever), size=num_immigrants, replace=False)
    stay_single_forever -= set(marry_strangers)

    unions |= {(spouse, immigrant) for spouse, immigrant in zip(marry_strangers, immigrants)}

    marriage_distances.extend([-1] * num_immigrants)

    return unions, num_immigrants, marriage_distances, immigrants, wont_marry_until_next_time, len(stay_single_forever)
