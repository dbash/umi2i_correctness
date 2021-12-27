import os
import numpy as np
import tqdm
import argparse


CELEBA_ATTR_LIST = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 
        'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
        'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
        'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
        'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes',
        'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
        'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling',
        'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
        'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
        ]


def format_line_dshapes(ln):
  str_arr = ln.split(" ")
  if "" in str_arr:
    str_arr.remove("")
  if "\n" in str_arr:
    str_arr.remove("\n")
  bname = str_arr[0]
  int_arr = np.array([int(x) for x in str_arr[1:]])
  return bname, int_arr


def format_line_celeba(ln):
  str_arr = ln.split(" ")
  bname = str_arr[0].replace(".jpg", ".png")
  int_arr = np.array([int(elem) for elem in str_arr[1:] if elem != '\n' and elem != '']) 
  return bname, int_arr


def format_line_synaction(ln):
  str_arr = ln.split(" ")
  bname = os.path.basename(str_arr[0])
  attr_arr = np.zeros(3)
  # for i in range(3):
  #   attr_arr[i] = int(str_arr[i+1])
  attr_arr = np.array([int(elem) for elem in str_arr[1:] if elem != '\n' and elem != ''])
  return bname, attr_arr


def format_line_poses(ln):
  str_arr = ln.split(" ")
  j = 0
  int_arr = np.zeros((26,))
  img_fname = str_arr[0].split("/")[-1]
  # print(str_arr)
  for elem in str_arr[1:]:
    if elem != '' and elem != '\n':
      int_arr[j] = int(elem)
      j += 1
  return img_fname, int_arr


def get_pose_dict(fpath):
  out_dict = {}
  pose_file = open(fpath, "r")
  for ln in pose_file:
    fname, attrs = format_line_poses(ln)
    out_dict[fname] = attrs
  pose_file.close()
  return out_dict


def get_shapes_specs():
    spec = {}
    spec["content_mask"] = np.array([0, 0, 1, 0, 1, 0, 0])
    spec["a_style_mask"] = np.array([1, 1, 0, 0, 0, 0, 0])
    spec["b_style_mask"] = np.array([0, 0, 0, 1, 0, 1, 0])
    spec["a_fixed_attrs"] = np.array([0, 0, 0, 5, 0, 0, 1])
    spec["b_fixed_attrs"] = np.array([0, 7, 0, 0, 0, 0, 0])
    spec["domain_attr_idx"] = -1
    return spec

def get_celeba_spec():
    spec = {}
    male_style_attributes = {"Young": 1, "Smiling": 1, "No_Beard": 1, 
                           "Sideburns": 0, "Mustache": 0, "Goatee": 0, 
                           'Heavy_Makeup': 1}
    female_style_attributes = {"Blond_Hair": 0, "Brown_Hair": 0, "Black_Hair": 1}
    content_attributes = {'5_o_Clock_Shadow', 'Arched_Eyebrows', 'Bags_Under_Eyes',
                            'Big_Lips', 'Big_Nose', 'Blurry', 'Bushy_Eyebrows', 'Chubby',
                            'Double_Chin', 'Eyeglasses', 'High_Cheekbones', 
                            'Narrow_Eyes', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
                            'Straight_Hair', 'Wavy_Hair', 'Wearing_Hat'}
    domain_attr = "Male"
    a_style_list = [CELEBA_ATTR_LIST.index(x) for x in list(male_style_attributes.keys())]

    b_style_list = [CELEBA_ATTR_LIST.index(x) for x in list(female_style_attributes.keys())]
    domain_attr_idx = CELEBA_ATTR_LIST.index(domain_attr)
    style_list = a_style_list + b_style_list 
    spec["domain_attr_idx"] = domain_attr_idx
    spec["a_style_mask"] = np.zeros(len(CELEBA_ATTR_LIST))
    spec["a_style_mask"][a_style_list] = 1.
    spec["b_style_mask"] = np.zeros(len(CELEBA_ATTR_LIST))
    spec["b_style_mask"][b_style_list] = 1.
    content_list = np.array([CELEBA_ATTR_LIST.index(x) for x in list(content_attributes)])
    spec["content_mask"] = np.zeros(len(CELEBA_ATTR_LIST))
    spec["content_mask"][content_list] = 1.
    spec["a_fixed_attrs"] = np.zeros(len(CELEBA_ATTR_LIST))
    spec["b_fixed_attrs"] = np.zeros(len(CELEBA_ATTR_LIST))
    for attr, val in female_style_attributes.items():
        spec["a_fixed_attrs"][CELEBA_ATTR_LIST.index(attr)] = val
    spec["a_fixed_attrs"][domain_attr_idx] = 1.
    for attr, val in male_style_attributes.items():
        spec["b_fixed_attrs"][CELEBA_ATTR_LIST.index(attr)] = val
    spec["b_fixed_attrs"][domain_attr_idx] = 0.
    return spec


def get_synaction_spec():
    spec = {}
    spec["content_mask"] = np.array([0, 0, 1])
    spec["a_style_mask"] = np.array([0, 1, 0])
    spec["b_style_mask"] = np.array([1, 0, 0])

    spec["a_fixed_attrs"] = np.array([0, 0, 0])
    spec["b_fixed_attrs"] = np.array([0, 0, 0])
    spec["domain_attr_idx"] = 0
    return spec


def get_spec(dataset: str, ignore_splitting_attr: bool = False):
    if "shapes" in dataset:
        spec = get_shapes_specs()
    elif "celeba" in dataset:
        spec = get_celeba_spec()
    elif "synaction" in dataset:
        spec = get_synaction_spec()
    else:
        raise ValueError("Unknown dataset type %s" % dataset)   
    spec["content_list"] = np.where(spec["content_mask"] > 0)[0]
    spec["a_style_list"] = np.where(spec["a_style_mask"] > 0)[0]
    spec["b_style_list"] = np.where(spec["b_style_mask"] > 0)[0]
    a_fixed_list = spec["b_style_list"].copy() 
    b_fixed_list = spec["a_style_list"].copy() 
    if not ignore_splitting_attr:
        a_fixed_list = list(a_fixed_list) + [spec["domain_attr_idx"],]
        b_fixed_list = list(b_fixed_list) + [spec["domain_attr_idx"],]
    spec["a_fixed_list"] = a_fixed_list
    spec["b_fixed_list"] = b_fixed_list
    return spec


def print_to_file(fl, str_line):
    print(str_line)
    fl.write(str_line)


def compute_all_metrics(original_attr_dict, trans_attr_dict, out_fname, exp_spec):
    epsilon = 1e-10 # div-by-0 handling
    # unpacking specs
    attr_names = exp_spec["attributes_names"]
    n_attributes = len(attr_names)
    synaction = "synaction" in exp_spec["dataset"]
    a_style_mask = exp_spec["a_style_mask"]
    b_style_mask = exp_spec["b_style_mask"]
    b_fixed_attrs = exp_spec["b_fixed_attrs"]
    a_fixed_attrs = exp_spec["a_fixed_attrs"]
    content_mask = exp_spec["content_mask"]
    b_style_list = exp_spec["b_style_list"]
    a_style_list = exp_spec["a_style_list"]
    a_fixed_list = exp_spec["a_fixed_list"]
    b_fixed_list = exp_spec["b_fixed_list"]
    content_list = exp_spec["content_list"]
    domain_attr_idx = exp_spec["domain_attr_idx"]
    domain_attr_value = exp_spec["domain_attr_value"]

    def compute_specific_attributes(
        content_attrs, style_attrs, trans_attrs, a2b=True, synaction=False):
        if a2b:
            fixed_idx = b_fixed_list 
            style_idx = b_style_list 
            fixed_attrs = b_fixed_attrs
        else:
            fixed_idx = a_fixed_list
            style_idx =  a_style_list
            fixed_attrs = a_fixed_attrs

        trans_count = np.array([content_attrs[x] != fixed_attrs[x] for x in fixed_idx])
        trans_q = np.array([trans_attrs[x] == fixed_attrs[x] for x in fixed_idx]) * trans_count
        bias_count = content_attrs == style_attrs


        if synaction:
            content_count = np.array([1])
            bias_count[-1] = 0
        else:
            content_count = np.array([content_attrs[x] != style_attrs[x] for x in content_list])

        content_q = np.array([content_attrs[x] == trans_attrs[x] for x in content_list]) * content_count

        style_count = np.array([content_attrs[x] != style_attrs[x] for x in style_idx]) 
        style_q = np.array([trans_attrs[x] == style_attrs[x] for x in style_idx]) * style_count

        bias_mtr = (trans_attrs != content_attrs) * bias_count
        return {"translation": (trans_q, trans_count),
                "content": (content_q, content_count),
                "style": (style_q, style_count),
                "bias": (bias_mtr, bias_count)}

    # initializing counters for each attribute
    a2b_style_attrs = np.zeros(len(b_style_list))
    b2a_style_attrs = np.zeros(len(a_style_list))
    a2b_style_count = epsilon * np.ones_like(a2b_style_attrs)
    b2a_style_count = epsilon * np.ones_like(b2a_style_attrs)
    content_pres_attrs = np.zeros(len(content_list))
    content_count = epsilon * np.ones_like(content_pres_attrs)
    bias_attributes = np.zeros(n_attributes)
    bias_count = epsilon * np.ones(n_attributes)
    
    transfer_attributes = np.zeros(n_attributes)
    transfer_count = np.zeros(n_attributes)
    # overall metrics
    trans_q_a2b = np.zeros(len(b_fixed_list))
    trans_q_b2a = np.zeros(len(a_fixed_list))
    trans_count_a2b = epsilon * np.ones_like(trans_q_a2b)
    trans_count_b2a = epsilon * np.ones_like(trans_q_b2a)
    num_a2b = 0
    num_b2a = 0
    # computing metrics
    for fname, attrs in trans_attr_dict.items():
        content_fname = fname.split("_")[0] + ".png"
        style_fname = fname.split("_")[1] + ".png"
        content_attrs = original_attr_dict[content_fname].copy()
        style_attrs = original_attr_dict[style_fname].copy()
        if synaction:
            style_attrs[-1] = 0
        a2b = content_attrs[domain_attr_idx] == domain_attr_value
        metrics_dict = compute_specific_attributes(
            content_attrs, style_attrs, attrs, a2b, synaction=synaction)

        content_pres_attrs += metrics_dict["content"][0]
        content_count += metrics_dict["content"][1]
        bias_attributes += metrics_dict["bias"][0]
        bias_count += metrics_dict["bias"][1]
        
        if a2b:
            a2b_style_attrs += metrics_dict["style"][0] 
            a2b_style_count += metrics_dict["style"][1]
            trans_q_a2b += metrics_dict["translation"][0]
            trans_count_a2b += metrics_dict["translation"][1]
            num_a2b += 1
            
            perfect_dis_attrs = style_attrs * b_style_mask + b_fixed_attrs * a_style_mask + content_attrs * content_mask
            perfect_dis_attrs[domain_attr_idx] = style_attrs[domain_attr_idx]
        else:
            b2a_style_attrs += metrics_dict["style"][0] 
            b2a_style_count += metrics_dict["style"][1]
            trans_q_b2a += metrics_dict["translation"][0]
            trans_count_b2a += metrics_dict["translation"][1]
            num_b2a += 1
            perfect_dis_attrs = style_attrs * a_style_mask + a_fixed_attrs * b_style_mask + content_attrs * content_mask
            
            diff_mask = content_attrs != style_attrs
            transfer_count += diff_mask
            transfer_attributes += (perfect_dis_attrs == attrs) * diff_mask
    
    # aggregating the metrics
    N = len(trans_attr_dict)
    content_pres_attrs /= content_count
    bias_attributes /= bias_count
    transfer_attributes /= transfer_count + epsilon
    a2b_style_attrs /= a2b_style_count
    b2a_style_attrs /= b2a_style_count
    trans_q_a2b /= trans_count_a2b
    trans_q_b2a /= trans_count_b2a
    a2b_style_metric = np.sum(a2b_style_attrs) / np.sum(a2b_style_count > epsilon)
    b2a_style_metric = np.sum(b2a_style_attrs) / np.sum(b2a_style_count > epsilon)
    a2b_trans_metric = np.sum(trans_q_a2b) / np.sum(trans_count_a2b > epsilon)
    b2a_trans_metric = np.sum(trans_q_b2a) / np.sum(trans_count_b2a > epsilon)

    a2b_style_descr = {attr_names[b_style_list[x]]: "%.4f"% a2b_style_attrs[x] for x in range(len(b_style_list))}
    b2a_style_descr = {attr_names[a_style_list[x]]: "%.4f"% b2a_style_attrs[x] for x in range(len(a_style_list))}
    a2b_trans_descr = {attr_names[b_fixed_list[x]]: "%.4f"% trans_q_a2b[x] for x in range(len(b_fixed_list))}
    b2a_trans_descr = {attr_names[a_fixed_list[x]]: "%.4f"% trans_q_b2a[x] for x in range(len(a_fixed_list))}
    content_descr = {attr_names[content_list[x]]: "%.4f"% content_pres_attrs[x] for x in range(len(content_list))}
    bias_descr = {attr_names[x]: "%.4f"% bias_attributes[x] for x in range(n_attributes)}
    # writing results to file
    with open(out_fname, "w") as fl:
        print_to_file(fl, "Results for % s:\n" % exp_spec["dataset"])
        print_to_file(fl, "Q_trans = %.4f \n" % np.mean([a2b_trans_metric, b2a_trans_metric]))
        print_to_file(fl, "Content Preservation = %.4f \n" % np.mean(content_pres_attrs))
        print_to_file(fl, "A2B Style translation = %.4f \n" % a2b_style_metric)
        print_to_file(fl, "B2A Style translation = %.4f \n" % b2a_style_metric)
        print_to_file(fl, "Style translation = %.4f \n" % np.mean([a2b_style_metric, 
                                                                b2a_style_metric]))
        print_to_file(fl, "Model Bias = %.4f \n" % np.mean(bias_attributes))
        print_to_file(fl, "Disentanglement quality = %.4f \n" % np.mean([np.mean([a2b_style_metric, 
                                                                b2a_style_metric]), np.mean(content_pres_attrs)]))
        print_to_file(fl, "Correct attribute change = %.4f \n" % np.mean(transfer_attributes))

        print_to_file(fl, "Detailed results: \n")
        print_to_file(fl, "A2B Style manipulation quality = %s \n" % str(a2b_style_descr))
        print_to_file(fl, "B2A Style manipulation quality = %s \n" % str(b2a_style_descr))
        print_to_file(fl, "Content preservation quality = %s \n" % str(content_descr))
        print_to_file(fl, "A2B Translation quality = %s \n" % str(a2b_trans_descr))
        print_to_file(fl, "B2A Translation quality = %s \n" % str(b2a_trans_descr))
        print_to_file(fl, "Bias stats = %s \n" % str(bias_descr))


def read_attributes(attr_file, format_line_fn): 
    """Reads attribute prediction file to dictionary."""
    out_dict = {}
    with open(attr_file, "r") as fl:
        for i, ln in enumerate(fl.readlines()):
            bname, attrs = format_line_fn(ln)
            out_dict[bname] = attrs
    return out_dict


def compute_metrics(
    dataset: str, trans_attr_file: str, original_attr_file: str, out_file: str):
    if "shapes" in dataset:
        attributes_list = [
            'floor_hue', 'wall_hue', 'object_hue', 'size',
            'shape', 'orientation', 'domain']
        format_line_method = format_line_dshapes
        domain_attr_idx = -1
        domain_attr_value = 1
        ignore_splitting_attr = True
    elif "celeba" in dataset:
        attributes_list = CELEBA_ATTR_LIST
        format_line_method = format_line_celeba
        domain_attr_idx = attributes_list.index("Male")
        domain_attr_value = 1
        ignore_splitting_attr = False
    elif "synaction" in dataset:
        attributes_list = ["Identity", "Background", "Pose_content"]
        ignore_splitting_attr = True
        domain_attr_value = 0
        format_line_method = format_line_synaction
        domain_attr_idx = attributes_list.index("Identity")
    else:
        raise ValueError("Unknown dataset type %s" % dataset)
    exp_spec = get_spec(dataset, ignore_splitting_attr)
    exp_spec["attributes_names"] = attributes_list
    exp_spec["domain_attr_value"] = domain_attr_value
    exp_spec["domain_attr_idx"] = domain_attr_idx
    exp_spec["dataset"] = dataset 
    original_attr_dict = read_attributes(
        original_attr_file, format_line_method)
    trans_attr_dict = read_attributes(
        trans_attr_file, format_line_method)
    compute_all_metrics(
        original_attr_dict, trans_attr_dict, out_file, exp_spec)
        


def main():
    parser = argparse.ArgumentParser(description='Compute image translation metrics.')
    parser.add_argument('--method_attr_file',  type=str, default="./pred_attributes.txt",
                        help='file containing attribute predictions for each'
                        'translation example.')
    parser.add_argument('--out_file', type=str, default="./metrics.txt",
                        help='Output metrics file path')
    parser.add_argument('--original_attr_file', type=str, default="./original_attributes.txt",
                        help='file containing attribute predictions for each'
                        'example from the original dataset.')
    parser.add_argument('--dataset', type=str, choices=["shapes", "celeba", "synaction"],
                        default="shapes",
                        help='dataset type (shapes, celeba or synaction).')
    
    args = parser.parse_args()
    compute_metrics(
        args.dataset,
        args.method_attr_file,
        args.original_attr_file,
        args.out_file)


if __name__ == "__main__":
    main()