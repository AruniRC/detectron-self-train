import os
import sys
import cv2
import numpy as np

def collect_vid(folder,save_folder):
    vid_list = {}
    for img in sorted(os.listdir(folder)):
        vid_id = int(img.split('_')[1])
        frame = cv2.imread(os.path.join(folder,img))
        print(img)
        if vid_id in vid_list.keys():
            vid_list[vid_id].append(frame)
        else:
            vid_list[vid_id] = [frame]
        ext = '.mov'
        max_len = 30
        if len(vid_list[vid_id]) == max_len:
            save_vid_file = os.path.join(save_folder,str(vid_id)+ext)
            print('Saving',save_vid_file,len(vid_list))
            save_vid(save_vid_file,vid_list[vid_id])
            del(vid_list[vid_id])
    return vid_list.items()
    
def save_vid(vid_file_path,vid):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_size = (vid[0].shape[1],vid[0].shape[0])
    print(out_size,len(vid))
    out = cv2.VideoWriter(vid_file_path,fourcc,17,out_size)
    for fid,frame in enumerate(vid):
        if frame is None:
            print('>>> BAD:',vid_file_path,fid)
            continue
        frame = np.uint8(frame)
        out.write(frame)
    #out.close()

def cityscapes_im_seq2vid(seq_folder,save_folder):
    #ext = '.mp4'
    for subdir in os.listdir(seq_folder)[2:]:
        subdir_folder = os.path.join(seq_folder,subdir)
        save_subdir = os.path.join(save_folder,subdir)
        os.system('mkdir '+save_subdir)
        for city in os.listdir(subdir_folder)[::-1]:
            city_folder = os.path.join(subdir_folder,city)
            save_city = os.path.join(save_subdir,city)
            os.system('mkdir '+save_city)
            vid_list = collect_vid(city_folder,save_city)
            #for vid_id,vid in vid_list:
            #    vid_file = os.path.join(save_city,str(vid_id)+ext)
            #    save_vid(vid_file,vid)
            #    print('Saving',save_vid)

if __name__ == '__main__':
    seq_folder = '/mnt/nfs/work1/elm/hzjiang/Data/CityScapes/leftImg8bit_sequence/'
    save_folder = '/mnt/nfs/scratch1/pchakrabarty/cityscapes/vid/leftImg8bit_sequence/'
    cityscapes_im_seq2vid(seq_folder,save_folder)
