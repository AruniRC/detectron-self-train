import os

#src = '/mnt/nfs/scratch1/pchakrabarty/cityscapes/vid/leftImg8bit_sequence/train'
#target = '/mnt/nfs/scratch1/pchakrabarty/cityscapes/vid_dump/train'

#src = '/mnt/nfs/scratch1/pchakrabarty/cityscapes/vid/leftImg8bit_sequence/val'
#target = '/mnt/nfs/scratch1/pchakrabarty/cityscapes/vid_dump/val'

#src = '/mnt/nfs/scratch1/ashishsingh/FALL2018/Detectron-pytorch-video/Outputs/detections_videos/frcnn-R-50-C4-1x/cityscapes/train-BDD_PED_kitti_cityscapes_val-video_conf-0.50/car'
#target = '/mnt/nfs/scratch1/pchakrabarty/cityscapes/det_results'

#src = '/mnt/nfs/scratch1/ashishsingh/FALL2018/MDNet-tracklet/Output/cityscapes_hp/car/tracklet_length_5'
#target = '/mnt/nfs/scratch1/pchakrabarty/cityscapes/tracklet_len5'

src = '/mnt/nfs/scratch1/ashishsingh/FALL2018/MDNet-tracklet/Output/cityscapes_hp/car/tracklet_length_3'
target = '/mnt/nfs/scratch1/pchakrabarty/cityscapes/tracklet_len3'

for city in os.listdir(src):
    #for filename in os.listdir(os.path.join(src,city)):
    for filename in os.listdir(os.path.join(src,city,'hp-res')):
        #src_file = os.path.join(src,city,filename)
        src_file = os.path.join(src,city,'hp-res',filename)
        target_file = os.path.join(target,city+'-'+filename)
        print(src_file,'==>',target_file)
        os.system('cp '+src_file+' '+target_file)
        
        # Edit the fddb file, if we're flattening an fddb dump folder
        with open(target_file,'r') as f:
            t = f.readlines()
        f.close()
        t = ['/'.join([l.split('/')[0]]+[city+'-'+g for g in l.split('/')[1:]]) if l.startswith('frame') else l for l in t]
        with open(target_file,'w') as f:
            for g in t:
                f.write(g)
        f.close()
