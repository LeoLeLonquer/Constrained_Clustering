import collections

def must_link_fusion(cons, out_type=None):
    ml_groups = collections.defaultdict(set) # pt => list of pts
    ml_labels = dict() # pt => nb
    for pt1, pt2 in cons:
        ml_groups[pt1].add(pt2)
        ml_groups[pt1].add(pt1)
        ml_groups[pt2].add(pt1)
        ml_groups[pt2].add(pt2)
        ml_labels[pt1] = pt1
        ml_labels[pt2] = pt2

    pts = set(ml_groups.keys())
    for pt in pts:
        for pt2 in pts :
            # if the intersection is not empty
            if ml_labels[pt] != ml_labels[pt2] \
            and not not ml_groups[ml_labels[pt]].intersection(ml_groups[ml_labels[pt2]]):
                ml_groups[ml_labels[pt]].update(ml_groups[ml_labels[pt2]])
                ml_labels[pt2] = pt
                del ml_groups[pt2]

    if out_type == 'values':
        out = list(ml_groups.values())
    else :
        out = ml_groups
    return out, ml_labels

def clean(n_points, cons):
    """ converts links btw 1 and -1 to 1 and n_points-1"""
    cons_to_modify = filter(lambda pts: pts[0] < 0 or pts[1] < 0, cons)
    new_cons = list(filter(lambda pts: pts[0] >= 0 and pts[1] >= 0, cons))

    for pt1, pt2 in cons_to_modify:
        if pt1 < 0:
            pt1 = n_points + pt1
        if pt2 < 0:
            pt2 = n_points + pt2
        new_cons.append((pt1,pt2))
    return new_cons

def propagate(number_of_points, ml_cons, cl_cons):
    """ returns :
        - ordered_ml_groups a tab where index == label => group of points
        - new_cl_cons a dic where label => set of label
    """
    ml_cons = clean(number_of_points, ml_cons)
    cl_cons = clean(number_of_points, cl_cons)

    ml_groups, ml_labels = must_link_fusion(ml_cons)
    new_cl_cons = collections.defaultdict(set) # label => set of labels
    ordered_ml_groups = [] #index == label => set of points
    mapped_values = set({}) # values in ml_groups

    # listing of ml_groups in range(0, len(ml_groups))
    label_nb = -1
    for index, values in enumerate(ml_groups.values()):
        label_nb = label_nb + 1
        ordered_ml_groups.append(values)
        mapped_values.update(values)
        for pt in values:
            ml_labels[pt] = index

    for pt in range(number_of_points):
        if pt in mapped_values:
            pass
        else:
            label_nb = label_nb + 1
            ml_labels[pt] = label_nb
            ordered_ml_groups.append(set({pt}))

    # fusion of cl_cons
    for pt1, pt2 in cl_cons:
        new_cl_cons[ml_labels[pt1]].add(ml_labels[pt2])
        new_cl_cons[ml_labels[pt2]].add(ml_labels[pt1])

    return ordered_ml_groups, new_cl_cons

