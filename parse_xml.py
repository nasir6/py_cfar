from xml.dom.minidom import parse, parseString
import glob
import sys

data_dir = '/media/nasir/Drive1/code/SAR/python_cfar/SAR-Ship-Dataset'
paths = glob.glob(f"{data_dir}/Annotations_new/*.xml")

for index in range(0, len(paths)):
    path = paths[index]
    gt_path = path.replace('Annotations_new', 'ground-truth').replace('.xml', '.txt')
    with open(gt_path, 'w') as f1:
        dom1 = parse(path)  # parse an XML file by name

        itemlist = dom1.getElementsByTagName('object')
        bboxes = []
        for item in itemlist:
            bbox_node = item.getElementsByTagName('bndbox')
            xmin = bbox_node[0].getElementsByTagName('xmin')[0].firstChild.nodeValue
            ymin = bbox_node[0].getElementsByTagName('ymin')[0].firstChild.nodeValue
            xmax = bbox_node[0].getElementsByTagName('xmax')[0].firstChild.nodeValue
            ymax = bbox_node[0].getElementsByTagName('ymax')[0].firstChild.nodeValue
            f1.write(f"Ship {int(xmin)} {int(ymin)} {int(xmax) - int(xmin)} {int(ymax) - int(ymin)}\n")
            bboxes.append([int(xmin), int(ymin), int(xmax) - int(xmin), int(ymax) - int(ymin)])
        
    sys.stdout.write(f"\r {index} / {len(paths)} , {path}")
    sys.stdout.flush() 
