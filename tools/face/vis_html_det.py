mport os.path as osp
from html4vision import Col, imagetable


# table description
cols = [
    Col('id1', 'ID'), # make a column of 1-based indices
    Col('img', 'vgg16', 'det_vgg16_Flickr_vis-0.70/*.jpg'), # specify image content for column 2

]

imagetable(cols, outfile='det_vgg16_Flickr_vis-0.70.html', title='Flickr conf pert-adapt', 
            style=None)

