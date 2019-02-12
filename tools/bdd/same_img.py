import json
import collections


if __name__ == '__main__':
    json1 = 'data/CS6_annot/cs6-train-gt.json'
    json2 = 'data/CS6_annot/cs6-train-hp.json'

    save1 = 'cs6-train-gt_same_imgs.json'
    save2 = 'cs6-train-hp_same_imgs.json'
 
    with open(json1,'r') as f:
        j1 = json.load(f)
    f.close()
    with open(json2,'r') as f:
        j2 = json.load(f)
    f.close()


    print('Init num images:',len(j1['images']),len(j2['images']))
    print('Init num annots:',len(j1['annotations']),len(j2['annotations']))

    f1 = set([i['file_name'].strip() for i in j1['images']])
    f2 = set([i['file_name'].strip() for i in j2['images']])

    common = f1.intersection(f2)

    # new jsons
    t1 = j1.copy()
    t2 = j2.copy()

    t1['images'] = [i for i in t1['images'] if i['file_name'] in common]
    t2['images'] = [i for i in t2['images'] if i['file_name'] in common]

    t1_img_id_set = set([i['id'] for i in t1['images']])
    t2_img_id_set = set([i['id'] for i in t2['images']])

    t1['annotations'] = [a for a in t1['annotations'] if a['image_id'] in t1_img_id_set]
    t2['annotations'] = [a for a in t2['annotations'] if a['image_id'] in t2_img_id_set]

    print('New num images:',len(t1['images']),len(t2['images']))
    print('New num annots:',len(t1['annotations']),len(t2['annotations']))

    t1 = collections.OrderedDict(t1)
    t2 = collections.OrderedDict(t2)

    with open(save1,'w') as f:
        json.dump(t1,f)
    f.close()

    with open(save2,'w') as f:
        json.dump(t2,f)
    f.close()

