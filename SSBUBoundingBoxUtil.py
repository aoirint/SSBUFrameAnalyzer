
def fighters_info_bbox(fighter_num):
    assert fighter_num in (2, 3, 4), 'Not implemented'

    # FIXME: magic number
    if fighter_num == 2:
        return [
            # Fighter 1
            [ 245    ,560, 240,160 ],
            # Fighter 2
            [ 245+495,560, 240,160 ],
        ]
    elif fighter_num == 3:
        return [
            # Fighter 1
            [ 75    ,560, 240,160 ],
            # Fighter 2
            [ 75+416,560, 240,160 ],
            # Fighter 3
            [ 75+832,560, 240,160 ],
        ]
    elif fighter_num == 4:
        return [
            # Fighter 1
            [ 98    ,560, 240,160 ],
            # Fighter 2
            [ 98+272,560, 240,160 ],
            # Fighter 3
            [ 98+544,560, 240,160 ],
            # Fighter 4
            [ 98+816,560, 240,160 ],
        ]


def fighters_damage_bboxes(fighter_num):
    info_bboxes = fighters_info_bbox(fighter_num=fighter_num)

    ret = []
    for fighter_idx, bbox in enumerate(info_bboxes):
        left = bbox[0] + 85
        top = bbox[1] + 50

        ret.append([
            [ left   ,top, 35,55 ],
            [ left+30,top, 35,55 ],
            [ left+60,top, 35,55 ],
            [ left+97,top+28, 18,25 ],
        ])
    return ret

def fighters_name_bbox(fighter_num):
    info_bboxes = fighters_info_bbox(fighter_num=fighter_num)

    ret = []
    for fighter_idx, bbox in enumerate(info_bboxes):
        ret.append([ bbox[0]+105,bbox[1]+110, 120,16 ])
    return ret

def fighters_chara_bbox(fighter_num):
    info_bboxes = fighters_info_bbox(fighter_num=fighter_num)

    ret = []
    for fighter_idx, bbox in enumerate(info_bboxes):
        ret.append([ bbox[0]+10,bbox[1]+28, 110,110 ])
    return ret

def fighters_stock_bboxes(fighter_num, stock_num=3):
    info_bboxes = fighters_info_bbox(fighter_num=fighter_num)

    ret = []
    for fighter_idx, bbox in enumerate(info_bboxes):
        left = bbox[0] + 73
        top = bbox[1] + 131

        ret.append(
            [ [ left + 17*k, top, 16, 16, ] for k in range(stock_num) ]
        )
    return ret
