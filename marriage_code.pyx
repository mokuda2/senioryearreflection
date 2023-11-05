import itertools
import numpy as np
import random
from memory_profiler import profile

@profile
def add_marriage_edges(list people, list prev_people, int num_people, dict marriage_probs,
                       float prob_marry_immigrant, float prob_marry, double[:, ::1] D,
                       dict indices, list original_marriage_dist,
                       double tol=0, double eps=1e-7):
    print("I'M IN ADD MARRIAGE EDGES FUNCTION")
    cdef dict finite_marriage_probs = {}
    cdef float key
    cdef float val
    for key, val in marriage_probs.items():
        if key > 0:
            finite_marriage_probs[key] = val

    cdef list desired_finite_distances = []
    cdef float distance
    cdef float prob
    for distance, prob in finite_marriage_probs.items():
        if prob > 0:
            desired_finite_distances.append(distance)

    cdef float minimum_permissible_distance = -1
    cdef float k
    for k in original_marriage_dist:
        if k > -1:
            minimum_permissible_distance = min(minimum_permissible_distance, k)

    cdef list marriage_distances = []
    cdef set unions = set()

    cdef set people_set = set(people)
    cdef int next_person = num_people + 1
    cdef int num_inf_couples_to_marry = round(prob_marry_immigrant * len(people) / 2)
    cdef float num_finite_couples_to_marry = round(prob_marry * len(people) / 2)
    cdef set will_marry = set(random.sample(people_set, len(people_set) // 2))

    cdef list wont_marry_until_next_time = []
    cdef float node
    for node in people_set:
        if node not in will_marry:
            wont_marry_until_next_time.append(node)

    people_set |= will_marry | set(prev_people)

    cdef set possible_couples = set()
    cdef float man
    cdef float woman
    for man, woman in itertools.combinations(people_set, 2):
        possible_couples.add((man, woman))

    cdef float man2
    cdef float woman2
    for man2, woman2 in itertools.combinations(prev_people, 2):
        possible_couples.remove((man2, woman2))

    cdef dict possible_finite_couples = {}
    cdef float man3
    cdef float woman3
    for man3, woman3 in possible_couples:
        distance = D[indices[man3], indices[woman3]]
        if distance >= minimum_permissible_distance:
            possible_finite_couples[(man3, woman3)] = distance

#    print("LINE 66")
    cdef dict preferred_couples = {}
    cdef double[:] couple1
    cdef float distance1
    ckeys = np.array(list(possible_finite_couples.keys()))
    cvalues = np.array(list(possible_finite_couples.values()))
    cdef int n = len(ckeys)
    cdef int i2
    for i2 in range(n-1):
        couple1 = ckeys[i2]
        distance1 = cvalues[i2]
#    for couple1, distance1 in possible_finite_couples.items():
        if distance1 in set(original_marriage_dist) and distance1 in desired_finite_distances:
            preferred_couples[couple1] = distance1

    cdef dict other_couples = {}
    cdef double[:] couple2
    cdef float distance2
    ckeys2 = np.array(list(possible_finite_couples.keys()))
    cvalues2 = np.array(list(possible_finite_couples.values()))
    cdef int n2 = len(ckeys2)
    cdef int i3
    for i3 in range(n2-1):
        couple2 = ckeys2[i3]
        distance2 = cvalues2[i3]
#    for couple2, distance2 in possible_finite_couples.items():
        if couple2 not in preferred_couples:
            other_couples[couple2] = distance2

#    print("LINE 81")
    cdef int iter2 = 0
    cdef list dis_probs = []
    cdef list dis_probs_pre = []
    cdef list dis_probs2 = []
    cdef float dis_prob3
    cdef list dis_probs3 = []
    cdef float d
    cdef float d2
    cdef float least_bad_distance = float('inf')
    cdef float d3
    cdef float d4
    cdef float d5
    cdef float total_prob
    cdef float total_prob2
    cdef int man_idx
    cdef int woman_idx
    cdef dict possible_finite_couples2 = {}
    cdef float pair10
    cdef float distance10
    cdef dict preferred_couples2 = {}
    cdef float couple3
    cdef float distance3
    cdef dict other_couples2 = {}
    cdef float couple4
    cdef float distance4
    cdef set stay_single_forever = set()
    cdef float couple6
    cdef float man6

    while possible_finite_couples and iter2 < num_finite_couples_to_marry:
        if preferred_couples:
            possible_finite_couples_array = list(preferred_couples.keys())

            for d in preferred_couples.values():
                dis_prob = finite_marriage_probs.get(d, eps)
                if abs(dis_prob) < tol:
                    dis_prob = 0
                dis_probs_pre.append(dis_prob)
            total_prob = sum(dis_probs_pre)

            for d2 in dis_probs_pre:
                dis_probs2.append(d2 / total_prob)

            try:
                couple_index = possible_finite_couples_array[np.random.choice(len(preferred_couples), p=dis_probs)]
            except Exception as e:
                print(f'Error: {e}')

            couple = (couple_index[0], couple_index[1])
        else:
            possible_finite_couples_array = list(other_couples.keys())
            for d3 in other_couples.values():
                if d3 >= minimum_permissible_distance:
                    least_bad_distance = min(least_bad_distance, d3)
            for d4 in other_couples.values():
                if d == least_bad_distance:
                    dis_prob3 = 1
                else:
                    dis_prob3 = 0
                if abs(dis_prob3) < tol:
                    dis_prob3 = 0
                dis_probs3.append(dis_prob)

            total_prob2 = sum(dis_probs)
            for d5 in dis_probs3:
                dis_probs.append(d5 / total_prob)

            couple_index = possible_finite_couples_array[np.random.choice(len(other_couples), p=dis_probs)]
            couple = (couple_index[0], couple_index[1])

        unions.add(couple)
        man_idx = indices[couple[0]]
        woman_idx = indices[couple[1]]
        marriage_distances.append(int(D[man_idx, woman_idx]))

        for pair10, distance10 in possible_finite_couples.items():
            if couple[0] not in pair10 and couple[1] not in pair10:
                possible_finite_couples2[pair10] = distance10

        for couple3, distance3 in possible_finite_couples.items():
            if distance3 in set(original_marriage_dist) and distance3 in desired_finite_distances:
                preferred_couples2[couple3] = distance3

        for couple4, distance4 in possible_finite_couples.items():
            if couple4 not in preferred_couples2:
                other_couples2[couple4] = distance4

        iter2 += 1

    if iter2 == 0:
        stay_single_forever = will_marry | set(prev_people)
    else:
        for couple6 in possible_couples:
            for man6 in couple6:
                stay_single_forever.add(man3)

    cdef dict possible_inf_couples = {}
    cdef float man5
    cdef float woman5
    for man5, woman5 in itertools.combinations(stay_single_forever, 2):
        if D[indices[man5], indices[woman5]] == -1:
            possible_inf_couples[(man5, woman5)] = D[indices[man5], indices[woman5]]

    iter2 = 0
    cdef int man_idx2 = 0
    cdef int woman_idx2 = 0
    cdef double[:] pair20
    cdef float distance20
    cdef set stay_single_forever2 = set()
    cdef double[:] couple30
    cdef float man30
    cdef int n3
    cdef int n4
    cdef int num_immigrants
    for ss in range(10):
#    while possible_inf_couples and iter2 < num_inf_couples_to_marry:
        print("possible_inf_couples size:", len(possible_inf_couples))
        print("iter2:", iter2)
        print("num_inf_couples_to_marry:", num_inf_couples_to_marry)
        possible_inf_couples_array = np.array(list(possible_inf_couples.keys()))
        print("after possible_inf_couples_array")
        couple_index = np.random.choice(len(possible_inf_couples))  # draw uniformly
        print("after couple_index")
        couple = (possible_inf_couples_array[couple_index][0], possible_inf_couples_array[couple_index][1])
        print("after couple")

        unions.add(couple)
        print("unions:", unions)
        man_idx2 = indices[couple[0]]
        woman_idx2 = indices[couple[1]]
        marriage_distances.append(int(D[man_idx2, woman_idx2]))
        ckeys3 = np.array(list(possible_inf_couples.keys()))
        cvalues3 = np.array(list(possible_inf_couples.values()))
        n3 = len(ckeys3)
        for i4 in range(n3-1):
            pair20 = ckeys3[i4]
            distance20 = cvalues3[i4]
#        for pair20, distance20 in possible_inf_couples.items():
            if couple[0] not in pair20 and couple[1] not in pair20:
                possible_inf_couples[pair20] = distance20
        print("after first for loop")

        iter2 += 1
        cset = np.array(list(possible_couples))
        n4 = len(cset)
#        print("possible_couples:", possible_couples)
        for i5 in range(n4-1):
            couple30 = cset[i5]
#        for couple30 in possible_couples:
            for man30 in couple30:
                stay_single_forever2.add(man30)
        print("after second for loop")

    num_immigrants = num_inf_couples_to_marry - iter2
    num_immigrants = min(len(stay_single_forever2), num_immigrants)

    cdef list immigrants = []
    cdef int i50
    for i50 in range(next_person, next_person + num_immigrants):
        immigrants.append(i50)
    print("num_immigrants:", num_immigrants)
    print("np.random.choice(list(stay_single_forever2), size=num_immigrants, replace=False):", np.random.choice(list(stay_single_forever2), size=num_immigrants, replace=False))
    marry_strangers = np.random.choice(list(stay_single_forever2), size=num_immigrants, replace=False)
    print("marry_strangers:", marry_strangers)
    stay_single_forever2 -= set(marry_strangers)

    unions = set()
    cdef float spouse
    cdef float immigrant
    for spouse, immigrant in zip(marry_strangers, immigrants):
        unions.add((spouse, immigrant))

    marriage_distances.extend([-1] * num_immigrants)

    return unions, num_immigrants, marriage_distances, immigrants, wont_marry_until_next_time, len(stay_single_forever2)
