import traceback
import warnings
import numpy as np
import random
import cv2
import pandas as pd
import torch
import gc
import glob as glob_module
from glob import glob
import os
from PIL import Image, ImageFile, ImageEnhance

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 1000000000000

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
yx_MEAN = [0.485, 0.456, 0.406]
yx_STD = [0.229, 0.224, 0.225]
WHITE_BALANCE_LST = ["NF-10", "NF-26", "NF-20", "NF-19", "NF-17", "NF-1", "NF-27", "NF-25", "NF-6", "NF-8", "NF-5", "NF-9", "NF-29", "NF-4", "NF-28"]

# 这是一个字典，将特定的病变类型映射到整数标签。例如，'LN'（淋巴结）被映射为0，而其他多种类型的病变如'Spleen', 'Bone', 'Soft', 和 'Aden'都被映射为4（归类为"Other"）
lesion_to_label = {
    'normal': 0,
    'Other': 1,
}

def sample_random_transform_params(h, w, max_rotation=30, max_translate_frac=0.1, do_flip=True):
    """
    抽样一组随机变换参数（用于保证 image 与 mask 用同一参数）
    返回 dict 包含开关与数值：
    {
      'flip_h': bool,
      'flip_v': bool,
      'rotate_flag': bool,
      'angle': int,
      'translate_flag': bool,
      'tx': int,
      'ty': int,
    }
    """
    flip_h = (random.random() < 0.5) if do_flip else False
    flip_v = (random.random() < 0.5) if do_flip else False

    rotate_flag = random.random() < 0.5
    angle = int(random.choice(np.arange(-max_rotation, max_rotation + 1))) if rotate_flag else 0

    translate_flag = random.random() < 0.5
    tx = int(random.randint(-int(max_translate_frac * w), int(max_translate_frac * w))) if translate_flag else 0
    ty = int(random.randint(-int(max_translate_frac * h), int(max_translate_frac * h))) if translate_flag else 0

    return {
        "flip_h": flip_h,
        "flip_v": flip_v,
        "rotate_flag": rotate_flag,
        "angle": angle,
        "translate_flag": translate_flag,
        "tx": tx,
        "ty": ty,
    }

def random_transform_np(img_np, max_rotation=10, pad_value=255, params=None, is_mask=False):
    """
    对输入图像/掩码进行随机数据增强，若传入 params 则使用该组参数（保证 image/mask 对齐）。
    - img_np: numpy array (H,W) 或 (H,W,C)
    - pad_value: 填充值（对 mask 通常用 255 或 0）
    - params: 由 sample_random_transform_params 返回的 dict；若 None 则内部随机采样（向后兼容）
    - is_mask: 若 True，则在插值处使用最近邻，且 warp/crop/pad 时使用 pad_value 合理处理
    返回增强后的 numpy 数组（类型与输入一致）
    """
    if params is None:
        h, w = img_np.shape[:2]
        params = sample_random_transform_params(h, w, max_rotation=max_rotation)

    h, w = img_np.shape[:2]
    # 统一使用 nearest/linear 插值 control
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR

    # 翻转
    if params.get("flip_h"):
        img_np = cv2.flip(img_np, 0)  # NOTE: 原代码 flip 参数位置相反，你的代码里先 flip 0 再 flip 1（我保持行为一致）
    if params.get("flip_v"):
        img_np = cv2.flip(img_np, 1)

    # 随机旋转（若开启）
    if params.get("rotate_flag", False):
        angle = params.get("angle", 0)
        rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        if img_np.ndim == 3:
            border_val = tuple([int(pad_value)] * img_np.shape[2])
        else:
            border_val = int(pad_value)
        img_np = cv2.warpAffine(img_np, rot_mat, (w, h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)

    # 随机平移（若开启）
    if params.get("translate_flag", False):
        tx = params.get("tx", 0)
        ty = params.get("ty", 0)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        if img_np.ndim == 3:
            border_val = tuple([int(pad_value)] * img_np.shape[2])
        else:
            border_val = int(pad_value)
        img_np = cv2.warpAffine(img_np, M, (w, h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)

    return img_np

def load_lesions(data_dir, pid, lesion_height, lesion_width,
                 img_extn, is_training, num_lesions=1, split="train", mask_placeholder=255):
    """
    mask_placeholder: 当掩码文件缺失时使用的占位像素值（默认为255，表示全ROI；若想把缺失标注表示为"无标注"可设为0）
    返回: lesions, keys, lesions_label, roi_masks
    lesions: Tensor (N,3,H,W)
    roi_masks: Tensor (N,3,H,W) 归一化到 [0,1]
    """
    base_dir = os.path.join(data_dir, "0", pid)
    base_files = glob_module.glob(os.path.join(base_dir, f"*.{img_extn}"))
    base_files = [f for f in base_files if not f.endswith(f"_mask.{img_extn}")]

    if len(base_files) == 0:
        print(f"[Error] 病例 {pid} 在 0 期文件夹下没有找到任何 *.{img_extn} 图像")
        return torch.zeros((0, 3, lesion_height, lesion_width)), [], torch.LongTensor([]), torch.zeros(
            (0, 3, lesion_height, lesion_width))

    chosen = np.random.choice(base_files, num_lesions, replace=True)

    lesions = []
    roi_masks = []
    keys = []
    lesions_label = []
    phases = ["-1", "0", "1"]

    for file0 in chosen:
        basename = os.path.basename(file0)
        parts = basename.split('_', 1)
        if len(parts) != 2:
            print(f"[Warning] 文件名 {basename} 格式异常，跳过")
            continue
        suffix = '_' + parts[1]
        label_str = parts[1].split('.')[0]
        label = lesion_to_label.get(label_str)
        if label is None:
            print(f"[Error] 标签 {label_str} 未定义，跳过 {basename}")
            continue

        imgs = []
        mask_imgs = []
        phase_keys = []
        ok = True
        for ph in phases:
            folder = os.path.join(data_dir, ph, pid)
            ct_path = glob_module.glob(os.path.join(folder, f"*{suffix}"))
            ct_path = [c for c in ct_path if not c.endswith(f"_mask.{img_extn}")]
            if len(ct_path) != 1:
                ok = False
                break

            base_name = os.path.splitext(os.path.basename(ct_path[0]))[0]
            mask_path = os.path.join(folder, f"{base_name}_mask.{img_extn}")

            ct_img = np.asarray(Image.open(ct_path[0]).convert("L"))
            imgs.append(ct_img)
            phase_keys.append(base_name)

            if os.path.exists(mask_path):
                mask_img = np.asarray(Image.open(mask_path).convert("L"))
            else:
                print(f"[Warning] 找不到ROI掩码文件: {mask_path}，使用占位值 {mask_placeholder}")
                mask_img = np.ones_like(ct_img, dtype=np.uint8) * int(mask_placeholder)
            mask_imgs.append(mask_img)

        if not ok:
            continue

        # 堆叠成 H x W x 3
        lesion = np.stack(imgs, axis=-1)
        roi_mask = np.stack(mask_imgs, axis=-1)

        # resize 到目标尺寸（图像用线性插值，掩码用最近邻）
        if lesion.shape[:2] != (lesion_height, lesion_width):
            lesion = cv2.resize(lesion, (lesion_width, lesion_height), interpolation=cv2.INTER_LINEAR)
            roi_mask = cv2.resize(roi_mask, (lesion_width, lesion_height), interpolation=cv2.INTER_NEAREST)

        if is_training:
            # 关键：用同一组 params 对 lesion 与 roi_mask 进行增强，保证对齐
            params = sample_random_transform_params(lesion_height, lesion_width, max_rotation=30, max_translate_frac=0.1)
            lesion = random_transform_np(lesion, max_rotation=30, pad_value=0, params=params, is_mask=False)
            # 对掩码使用相同 params；pad_value 设为 mask_placeholder
            roi_mask = random_transform_np(roi_mask, max_rotation=30, pad_value=mask_placeholder, params=params, is_mask=True)

        # 归一化掩码到 0-1
        roi_mask = roi_mask.astype(np.float32) / 255.0

        lesions.append(lesion.astype(np.float32))
        roi_masks.append(roi_mask)
        keys.extend(phase_keys)
        lesions_label.append(label)

    if not lesions:
        print(f"[Error] 病例 {pid} 没有成功加载任何 lesion")
        return torch.zeros((0, 3, lesion_height, lesion_width)), [], torch.LongTensor([]), torch.zeros(
            (0, 3, lesion_height, lesion_width))

    arr = np.stack(lesions, axis=0)
    arr = arr / 255.0
    # 注意：外部需定义 yx_MEAN, yx_STD，且能与 (H,W,3) 广播兼容
    arr = (arr - yx_MEAN) / yx_STD
    lesions = torch.from_numpy(arr.transpose(0, 3, 1, 2)).float()

    mask_arr = np.stack(roi_masks, axis=0)
    roi_masks = torch.from_numpy(mask_arr.transpose(0, 3, 1, 2)).float()

    lesions_label = torch.LongTensor(lesions_label)

    return lesions, keys, lesions_label, roi_masks

def load_all_bags_with_masks(data_dir, bag_height, bag_width, img_extn, num_bags, split="train",
                             mask_placeholder=0, debug=True):
    """
    加载 data_dir 下以 '*-img.<img_extn>' 结尾的图像与对应 '*-mask.<img_extn>' 掩码。
    如果实际加载数量 < num_bags，则循环重复已加载的样本直到凑满 num_bags（重复是按已有顺序循环）。
    如果实际加载数量 == 0，则返回全0填充的数组（形状与 num_bags 保持一致），避免返回不一致类型。
    返回:
        bags: np.ndarray (num_bags, bag_height, bag_width, 3) dtype uint8
        masks: np.ndarray (num_bags, bag_height, bag_width) dtype float32 (0/1)
        keys:  np.ndarray (num_bags,) dtype object (字符串 key)
    mask_placeholder: 当掩码文件缺失时用于占位的像素值（0 表示无标注，255 表示全部 ROI）
    """
    # if debug:
    #     print(f"[DEBUG BL ROI] load_all_bags_with_masks called with data_dir={data_dir}, img_extn={img_extn}, num_bags={num_bags}")

    files = glob(os.path.join(data_dir, f"*-img.{img_extn}"))
    # if debug:
    #     print(f"[DEBUG BL ROI] Found {len(files)} '*-img.{img_extn}' files under {data_dir}")

    bags = []
    masks = []
    keys = []
    candidate_files = files.copy()
    missing_mask_count = 0
    mask_coverages = []
    loaded_count = 0

    # 随机抽取直到达到 num_bags 或候选为空
    while len(bags) < num_bags and candidate_files:
        file = random.choice(candidate_files)
        pid = os.path.basename(os.path.dirname(file))
        # if debug:
        #     print(f"[DEBUG BL ROI] Try file: {file} (PID: {pid})")

        try:
            # 白平衡替换（可选）
            if pid in WHITE_BALANCE_LST:
                wb_file = file.replace("BLData", "BLDataWB")
                if os.path.exists(wb_file):
                    # if debug:
                    #     print(f"[DEBUG BL ROI] Apply white-balance replacement: {file} -> {wb_file}")
                    file_to_load = wb_file
                else:
                    file_to_load = file
            else:
                file_to_load = file

            # 加载图像
            img = Image.open(file_to_load)
            rgb_img = img.convert("RGB")
            bag = np.asarray(rgb_img)  # H, W, 3
            # if debug:
            #     print(f"[DEBUG BL ROI] Loaded image {file_to_load}: shape={bag.shape}, dtype={bag.dtype}")

            # 调整尺寸（cv2 resize 参数为 (width, height)）
            if (bag_height, bag_width) != bag.shape[:2]:
                # if debug:
                #     print(f"[DEBUG BL ROI] Resize image {file_to_load} {bag.shape[:2]} -> {(bag_height, bag_width)}")
                bag = cv2.resize(bag, (bag_width, bag_height), interpolation=cv2.INTER_LINEAR)

            # 对应掩码路径（优先 BLData 原始掩码）
            mask_file = file_to_load.replace("BLDataWB", "BLData").replace('-img', '-mask')
            # if debug:
            #     print(f"[DEBUG BL ROI] Expect mask file: {mask_file}")

            # 加载掩码（灰度）
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                missing_mask_count += 1
                # if debug:
                #     print(f"[WARNING BL ROI] Mask not found for {file_to_load}. Using placeholder {mask_placeholder}.")
                # 创建占位掩码（以当前 bag 尺寸）
                mask = np.ones((bag.shape[0], bag.shape[1]), dtype=np.uint8) * int(mask_placeholder)
            # else:
            #     if debug:
            #         print(f"[DEBUG BL ROI] Loaded mask {mask_file}: shape={mask.shape}, unique_sample={np.unique(mask)[:10]}")

            # 二值化掩码（threshold > 1 -> 255）
            try:
                _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            except Exception:
                mask_uint8 = mask.astype(np.uint8)
                _, binary_mask = cv2.threshold(mask_uint8, 1, 255, cv2.THRESH_BINARY)
            mask = binary_mask  # uint8 0/255

            # resize 掩码到 bag 尺寸（nearest）
            if (bag_height, bag_width) != mask.shape[:2]:
                # if debug:
                #     print(f"[DEBUG BL ROI] Resize mask {mask.shape[:2]} -> {(bag_height, bag_width)} (nearest)")
                mask = cv2.resize(mask, (bag_width, bag_height), interpolation=cv2.INTER_NEAREST)
                mask = np.where(mask > 127, 255, 0).astype(np.uint8)


            # ====== 在这里进行随机增强（image + mask 用同一 params） ======
            if split == "train":
                try:
                    # 采样参数（你可以调整 max_rotation/max_translate_frac）
                    params = sample_random_transform_params(bag.shape[0], bag.shape[1],
                                                            max_rotation=30, max_translate_frac=0.1, do_flip=True)
                    # 对 image 用白色边界（pad_value=255）更常见于病理切片；如需黑边改为 0
                    bag_aug = random_transform_np(bag, max_rotation=30, pad_value=255, params=params, is_mask=False)
                    # 对 mask 使用相同 params，pad_value 使用 mask_placeholder（0 或 255）
                    mask_aug = random_transform_np(mask, max_rotation=30, pad_value=mask_placeholder,
                                                   params=params, is_mask=True)

                    # 覆盖原 bag/mask
                    bag = bag_aug
                    mask = mask_aug.astype(np.uint8)
                except Exception as e:
                    # 如果增强失败，退回原始（不要阻塞训练）
                    # 如果需要 debug，可以在这里打印 e
                    pass
            # ============================================================

            # 转为 float 0/1
            mask_float = (mask.astype(np.float32) / 255.0)
            mask_float = np.where(mask_float > 0.5, 1.0, 0.0).astype(np.float32)

            # 存储
            bags.append(bag)
            masks.append(mask_float)
            key = os.path.basename(file_to_load).replace(f"-img.{img_extn}", "")
            keys.append(key)
            loaded_count += 1

            pos_ratio = float(mask_float.mean())
            mask_coverages.append(pos_ratio)

            # if debug:
            #     unique, counts = np.unique(mask_float, return_counts=True)
            #     print(f"[DEBUG BL ROI] Loaded key={key} | bag_shape={bag.shape} | mask_shape={mask_float.shape} | mask_unique={list(unique)} | pos_ratio={pos_ratio:.4f}")

            # 防止重复选中
            if file in candidate_files:
                candidate_files.remove(file)

        except PermissionError:
            # if debug:
            #     print(f"[ERROR BL ROI] PermissionError for file: {file}")
            if file in candidate_files:
                candidate_files.remove(file)
        except Exception as e:
            # if debug:
            #     print(f"[ERROR BL ROI] Failed to load {file}: {e}")
            #     traceback.print_exc()
            if file in candidate_files:
                candidate_files.remove(file)

    # 如果完全没有加载到任何样本，返回全0填充（避免返回 torch.Tensor(1) 导致不一致）
    if len(bags) == 0:
        # if debug:
        #     print(f"[WARNING BL ROI] No valid bags loaded under {data_dir}. Returning zero-filled arrays of length {num_bags}.")
        bags_np = np.zeros((num_bags, bag_height, bag_width, 3), dtype=np.uint8)
        masks_np = np.zeros((num_bags, bag_height, bag_width), dtype=np.float32)
        keys_np = np.array([""] * num_bags, dtype=object)
        return bags_np, masks_np, keys_np

    # 如果加载数量 < num_bags，循环重复已加载样本直到达到 num_bags
    cur_n = len(bags)
    if cur_n < num_bags:
        needed = num_bags - cur_n
        # if debug:
        #     print(f"[DEBUG BL ROI] Loaded {cur_n} < requested {num_bags}, repeating existing {cur_n} samples {needed} time(s) to reach {num_bags}.")
        for i in range(needed):
            idx = i % cur_n
            # 深拷贝确保独立对象（虽然对 numpy array 不一定必须）
            bags.append(bags[idx].copy())
            masks.append(masks[idx].copy())
            keys.append(keys[idx])  # 保留相同 key（也可加后缀标识重复）
        # if debug:
        #     print(f"[DEBUG BL ROI] After repetition, total samples = {len(bags)}")

    # 若加载数量 > num_bags（极少情况），截断到前 num_bags
    if len(bags) > num_bags:
        # if debug:
        #     print(f"[DEBUG BL ROI] Loaded more ({len(bags)}) than requested ({num_bags}); truncating to {num_bags}")
        bags = bags[:num_bags]
        masks = masks[:num_bags]
        keys = keys[:num_bags]

    # 转为 numpy arrays
    bags_np = np.stack(bags, axis=0)
    masks_np = np.stack(masks, axis=0)
    keys_np = np.array(keys, dtype=object)

    # debug 汇总
    # if debug:
    #     print("========================================")
    #     print(f"[DEBUG BL ROI] Loaded {loaded_count} unique files / requested {num_bags} (after repeat/truncation).")
    #     print(f"[DEBUG BL ROI] Missing mask count (used placeholder): {sum(1 for k in keys if k == '') if False else 'unknown in debug'} (see per-sample logs).")
    #     if mask_coverages:
    #         mc = np.array(mask_coverages)
    #         print(f"[DEBUG BL ROI] Mask coverage (pos ratio) mean={mc.mean():.4f}, min={mc.min():.4f}, max={mc.max():.4f}")
    #         percentiles = np.percentile(mc, [0, 25, 50, 75, 100])
    #         print(f"[DEBUG BL ROI] Mask coverage percentiles (0/25/50/75/100): {percentiles}")
    #     print(f"[DEBUG BL ROI] Final shapes: bags={bags_np.shape}, masks={masks_np.shape}, keys_len={len(keys_np)}")
    #     print("========================================")

    return bags_np, masks_np, keys_np

def normalize_words_np(words_np):
    # (N_B, N_W*N_H, W_H, W_W, C)
    words_np = words_np.astype(float)
    words_np /= 255.0
    words_np -= MEAN
    words_np /= STD
    # (N_B, N_W*N_H, W_H, W_W, C) -> (N_B, N_W*N_H, C, W_H, W_W)
    words_np = words_np.transpose(0, 1, 4, 2, 3)
    return words_np

def bags_to_words(bags, word_height, word_width, pid):
    # bags: (N_B, B_H, B_W, C)
    num_bags, bag_height, bag_width, channel = bags.shape
    # (N_B, B_H, B_W, C) -> (N_B, B_H, N_W, W_W, C)
    words = np.reshape(bags, (num_bags, bag_height, -1, word_width, channel))
    # (N_B, B_H, N_W, W_W, C) -> (N_B, N_W, B_H, W_W, C)
    words = words.transpose(0, 2, 1, 3, 4)
    # (N_B, N_W, B_H, W_W, C) -> (N_B, N_W*N_H, W_H, W_W, C)
    words = np.reshape(words, (num_bags, -1, word_height, word_width, channel))

    words = normalize_words_np(words)
    words_torch = torch.from_numpy(words).float()
    del words
    gc.collect()

    return words_torch

def masks_to_words(masks, word_height, word_width, pid):
    # masks: (N_B, B_H, B_W)
    num_bags, bag_height, bag_width = masks.shape
    # (N_B, B_H, B_W) -> (N_B, B_H, N_W, W_W)
    words = np.reshape(masks, (num_bags, bag_height, -1, word_width))
    # (N_B, B_H, N_W, W_W) -> (N_B, N_W, B_H, W_W)
    words = words.transpose(0, 2, 1, 3)
    # (N_B, N_W, B_H, W_W) -> (N_B, N_W*N_H, W_H, W_W)
    words = np.reshape(words, (num_bags, -1, word_height, word_width))

    words_torch = torch.from_numpy(words).long()
    del words

    return words_torch

def ccd_sex(x):
    # 患者：性别
    x = str(x).strip()
    if x in ["男"]:
        return torch.FloatTensor([0.0, 1.0])
    elif x in ["女"]:
        return torch.FloatTensor([1.0, 0.0])
    else:
        raise NotImplementedError

def ccd_age(x, split_age=60):
    # 患者：年龄
    if x <= split_age:
        return torch.FloatTensor([1.0, 0.0])
    elif x > split_age:
        return torch.FloatTensor([0.0, 1.0])
    else:
        raise NotImplementedError

def ccd_yx_flag(x):
    yx_flag = [0.0] * 3
    x = int(x)
    if x >= 0: yx_flag[x] = 1.0
    return torch.FloatTensor(yx_flag)

def ccd_yx_necrosis(x):
    # 是否囊变坏死（0 没有  1 有 ）
    x = str(x).strip()
    if x == "0":
        return torch.FloatTensor([1.0, 0.0])  # 没有
    elif x == "1":
        return torch.FloatTensor([0.0, 1.0])  # 有
    elif x == "\\":
        return torch.FloatTensor([0.0, 0.0])  # 不确定或缺失数据
    else:
        raise NotImplementedError("Invalid input for lymph node metastasis")

def ccd_yx_extra_in(x):
    # 是否壁外浸润（0 没有  1 有 ）
    x = str(x).strip()
    if x == "0":
        return torch.FloatTensor([1.0, 0.0])  # 没有
    elif x == "1":
        return torch.FloatTensor([0.0, 1.0])  # 有
    elif x == "\\":
        return torch.FloatTensor([0.0, 0.0])  # 不确定或缺失数据
    else:
        raise NotImplementedError("Invalid input for lymph node metastasis")

def ccd_yx_lymph_node(x):
    # 是否区域淋巴结转移（0 没有  1 有 ）
    x = str(x).strip()
    if x == "0":
        return torch.FloatTensor([1.0, 0.0])  # 没有
    elif x == "1":
        return torch.FloatTensor([0.0, 1.0])  # 有
    elif x == "\\":
        return torch.FloatTensor([0.0, 0.0])  # 不确定或缺失数据
    else:
        raise NotImplementedError("Invalid input for lymph node metastasis")

def _is_missing_value(x):
    """
    统一判断缺失值：
    None / NaN / 空字符串 / '\\' / 'nan' / 'None'
    """
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass

    s = str(x).strip()
    if s == "" or s.lower() in ["nan", "none", "null"] or s == "\\":
        return True
    return False


# 病灶位置的映射字典
lesion_mapping = {
    "三角区": [1, 0, 0, 0, 0, 0, 0, 0],
    "侧壁": [0, 1, 0, 0, 0, 0, 0, 0],
    "后壁": [0, 0, 1, 0, 0, 0, 0, 0],
    "顶壁": [0, 0, 0, 1, 0, 0, 0, 0],
    "下壁": [0, 0, 0, 0, 1, 0, 0, 0],
    "前壁": [0, 0, 0, 0, 0, 1, 0, 0],
    "弥漫": [0, 0, 0, 0, 0, 0, 1, 0,],
    "多区域": [0, 0, 0, 0, 0, 0, 0, 1],

}
def ccd_yx_pos_lesion(lesion_str):
    """
    病灶位置：
    缺失时返回 8 维全 0 向量；
    多位置用 “、” 分隔时可叠加。
    """
    if _is_missing_value(lesion_str):
        return torch.FloatTensor([0.0] * 8)

    lesions = str(lesion_str).strip().split("、")
    result_vector = [0] * 8

    for lesion in lesions:
        lesion = lesion.strip()
        if lesion == "":
            continue
        if lesion in lesion_mapping:
            result_vector = [x + y for x, y in zip(result_vector, lesion_mapping[lesion])]
        else:
            # 未知类别也按缺失处理，避免中断
            return torch.FloatTensor([0.0] * 8)

    return torch.FloatTensor(result_vector)

# 形态的映射字典
morphology_mapping = {
    "菜花状": [1, 0, 0],
    "乳头状": [0, 1, 0],
    "丘地状": [0, 0, 1]
}
def ccd_yx_form(morphology_str):
    """
    形态：
    缺失时返回 3 维全 0 向量；
    多形态用 “、” 分隔时可叠加。
    """
    if _is_missing_value(morphology_str):
        return torch.FloatTensor([0.0] * 3)

    morphologies = str(morphology_str).strip().split("、")
    result_vector = [0] * 3

    for morphology in morphologies:
        morphology = morphology.strip()
        if morphology == "":
            continue
        if morphology in morphology_mapping:
            result_vector = [x + y for x, y in zip(result_vector, morphology_mapping[morphology])]
        else:
            # 未知类别也按缺失处理，避免中断
            return torch.FloatTensor([0.0] * 3)

    return torch.FloatTensor(result_vector)

def ccd_yx_ca(x):
    # 是否钙化
    x = str(x).strip()
    if x == "0":
        return torch.FloatTensor([1.0, 0.0])  # 没有
    elif x == "1":
        return torch.FloatTensor([0.0, 1.0])  # 有
    elif x == "\\":
        return torch.FloatTensor([0.0, 0.0])  # 不确定或缺失数据
    else:
        raise NotImplementedError("Invalid input for lymph node metastasis")

def ccd_yx_stalk(x):
    # 蒂（stalk）
    x = str(x).strip()
    if x == "0":
        return torch.FloatTensor([1.0, 0.0])  # 没有
    elif x == "1":
        return torch.FloatTensor([0.0, 1.0])  # 有
    elif x == "\\":
        return torch.FloatTensor([0.0, 0.0])  # 不确定或缺失数据
    else:
        raise NotImplementedError("Invalid input for lymph node metastasis")

def ccd_yx_border(x):
    # 边界
    x = str(x).strip()
    if x == "0":
        return torch.FloatTensor([1.0, 0.0])  # 没有
    elif x == "1":
        return torch.FloatTensor([0.0, 1.0])  # 有
    elif x == "\\":
        return torch.FloatTensor([0.0, 0.0])  # 不确定或缺失数据
    else:
        raise NotImplementedError("Invalid input for lymph node metastasis")

def ccd_yx_size(x, max_val=15.0):
    """
    大小：
    缺失时返回 [0, 0]
    """
    if _is_missing_value(x):
        return torch.FloatTensor([0.0, 0.0])

    try:
        x = float(x)
        x_norm = min(max(x, 0.0) / max_val, 1.0)
        return torch.FloatTensor([x_norm, 1.0 - x_norm])
    except Exception:
        return torch.FloatTensor([0.0, 0.0])

def ccd_yx_thickness(x, max_val=50.0):
    """
    壁厚：
    缺失时返回 [0, 0]
    """
    if _is_missing_value(x):
        return torch.FloatTensor([0.0, 0.0])

    try:
        x = float(x)
        x_norm = min(max(x, 0.0) / max_val, 1.0)
        return torch.FloatTensor([x_norm, 1.0 - x_norm])
    except Exception:
        return torch.FloatTensor([0.0, 0.0])

def ccd_yx_CT_A(x, max_val=200.0):
    """
    CT_A：
    缺失时返回 [0, 0]
    """
    if _is_missing_value(x):
        return torch.FloatTensor([0.0, 0.0])

    try:
        x = float(x)
        x_norm = min(max(x, 0.0) / max_val, 1.0)
        return torch.FloatTensor([x_norm, 1.0 - x_norm])
    except Exception:
        return torch.FloatTensor([0.0, 0.0])


def ccd_yx_CT_D(x, max_val=200.0):
    """
    CT_D：
    缺失时返回 [0, 0]
    """
    if _is_missing_value(x):
        return torch.FloatTensor([0.0, 0.0])

    try:
        x = float(x)
        x_norm = min(max(x, 0.0) / max_val, 1.0)
        return torch.FloatTensor([x_norm, 1.0 - x_norm])
    except Exception:
        return torch.FloatTensor([0.0, 0.0])


def ccd_yx_CT_V(x, max_val=200.0):
    """
    CT_V：
    缺失时返回 [0, 0]
    """
    if _is_missing_value(x):
        return torch.FloatTensor([0.0, 0.0])

    try:
        x = float(x)
        x_norm = min(max(x, 0.0) / max_val, 1.0)
        return torch.FloatTensor([x_norm, 1.0 - x_norm])
    except Exception:
        return torch.FloatTensor([0.0, 0.0])


def ccd_yx_num(x, max_val=15.0):
    """
    数目：
    普通数值 -> [归一化值, 1-归一化值, 0]
    多发 -> [0, 0, 1]
    缺失 -> [0, 0, 0]
    """
    if _is_missing_value(x):
        return torch.FloatTensor([0.0, 0.0, 0.0])

    x_str = str(x).strip()
    if x_str == "多发":
        return torch.FloatTensor([0.0, 0.0, 1.0])

    try:
        x = float(x)
        x_norm = min(max(x, 0.0) / max_val, 1.0)
        return torch.FloatTensor([x_norm, 1.0 - x_norm, 0.0])
    except Exception:
        return torch.FloatTensor([0.0, 0.0, 0.0])

def ccd_bl_flag(x):
    bl_flag = [0.0] * 3
    x = int(x)
    if x >= 0: bl_flag[x] = 1.0
    return torch.FloatTensor(bl_flag)

def ccd_bl_Tis(x):
    # 是否存在原位癌Tis
    x = str(x).strip()
    if x == "0":
        return torch.FloatTensor([1.0, 0.0])  # 没有
    elif x == "1":
        return torch.FloatTensor([0.0, 1.0])  # 有
    elif x == "\\":
        return torch.FloatTensor([0.0, 0.0])  # 不确定或缺失数据
    else:
        raise NotImplementedError("Invalid input for lymph node metastasis")

def ccd_bl_T(x):
    # 组织学分级 (grade)
    x = str(x).strip()
    if x == "0":
        return torch.FloatTensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 低级别
    elif x == "1":
        return torch.FloatTensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # 中级别
    elif x == "2":
        return torch.FloatTensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # 高级别
    elif x == "3":
        return torch.FloatTensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # 中级别
    elif x == "4":
        return torch.FloatTensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])  # 高级别
    elif x == "5":
        return torch.FloatTensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # 高级别
    elif x == "\\":
        return torch.FloatTensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 缺失数据
    else:
        raise NotImplementedError("Invalid input for T")

def ccd_bl_N(x):
    # 组织学分级 (grade)
    x = str(x).strip()
    if x == "0":
        return torch.FloatTensor([1.0, 0.0, 0.0, 0.0])  # 低级别
    elif x == "1":
        return torch.FloatTensor([0.0, 1.0, 0.0, 0.0])  # 中级别
    elif x == "2":
        return torch.FloatTensor([0.0, 0.0, 1.0, 0.0])  # 高级别
    elif x == "3":
        return torch.FloatTensor([0.0, 0.0, 0.0, 1.0])  # 中级别
    elif x == "\\":
        return torch.FloatTensor([0.0, 0.0, 0.0, 0.0])  # 缺失数据
    else:
        raise NotImplementedError("Invalid input for N")

def ccd_bl_M(x):
    # 组织学分级 (grade)
    x = str(x).strip()
    if x == "0":
        return torch.FloatTensor([1.0, 0.0])  # 低级别
    elif x == "1":
        return torch.FloatTensor([0.0, 1.0])  # 中级别
    elif x == "\\":
        return torch.FloatTensor([0.0, 0.0])  # 缺失数据
    else:
        raise NotImplementedError("Invalid input for M")

def ccd_bl_qfjc(x):
    # 是否侵犯肌层（0 没有  1 有 ）
    x = str(x).strip()
    if x == "0":
        return torch.FloatTensor([1.0, 0.0])  # 没有
    elif x == "1":
        return torch.FloatTensor([0.0, 1.0])  # 有
    elif x == "\\":
        return torch.FloatTensor([0.0, 0.0])  # 不确定或缺失数据
    else:
        raise NotImplementedError("Invalid input for lymph node metastasis")

def ccd_bl_grade(x):
    # 组织学分级 (grade)
    x = str(x).strip()
    if x == "0":
        return torch.FloatTensor([1.0, 0.0, 0.0])  # 低级别
    elif x == "1":
        return torch.FloatTensor([0.0, 1.0, 0.0])  # 中级别
    elif x == "2":
        return torch.FloatTensor([0.0, 0.0, 1.0])  # 高级别
    elif x == "\\":
        return torch.FloatTensor([0.0, 0.0, 0.0])  # 缺失数据
    else:
        raise NotImplementedError("Invalid input for histological grade")

def ccd_bl_AJCC(x):
    # AJCC 分期
    stage_map = {
        "0":  [1, 0, 0, 0, 0, 0],
        "1": [0, 1, 0, 0, 0, 0],
        "2":   [0, 0, 1, 0, 0, 0],
        "3":  [0, 0, 0, 1, 0, 0],
        "4": [0, 0, 0, 0, 1, 0],
        "5":  [0, 0, 0, 0, 0, 1],
        "\\":  [0, 0, 0, 0, 0, 0]  # 缺失数据标记
    }

    x = str(x).strip()

    if x in stage_map:
        return torch.FloatTensor(stage_map[x])  # 返回对应的向量
    else:
        raise NotImplementedError(f"Invalid input for AJCC stage: {x}")


def ccd_Urine_PH(ph):
    try:
        ph_value = float(ph)
        return torch.FloatTensor([ph_value, 1])  # 有效值，标记为 0
    except ValueError:
        if ph == "\\":  # 判断是否为无效值
            return torch.FloatTensor([0.0, 0])  # 无效值，标记为 1
        else:
            raise NotImplementedError("Invalid input for ph value")

def ccd_NLR(ph):
    try:
        ph_value = float(ph)
        return torch.FloatTensor([ph_value, 1])  # 有效值，标记为 0
    except ValueError:
        if ph == "\\":  # 判断是否为无效值
            return torch.FloatTensor([0.0, 0])  # 无效值，标记为 1
        else:
            raise NotImplementedError("Invalid input for ph value")

def ccd_Urinary_infection(x):
    # 尿路感染
    x = str(x).strip()
    if x == "0":
        return torch.FloatTensor([1.0, 0.0])  # 没有
    elif x == "1":
        return torch.FloatTensor([0.0, 1.0])  # 有
    elif x == "\\":
        return torch.FloatTensor([0.0, 0.0])  # 不确定或缺失数据
    else:
        raise NotImplementedError("Invalid input for lymph node metastasis")

# 血尿的映射字典
hematuria_mapping = {
    "0": [1, 0, 0, 0, 0, 0, 0],
    "+-": [0, 1, 0, 0, 0, 0, 0],
    "1+": [0, 0, 1, 0, 0, 0, 0],
    "2+": [0, 0, 0, 1, 0, 0, 0],
    "3+": [0, 0, 0, 0, 1, 0, 0],
    "4+": [0, 0, 0, 0, 0, 1, 0],
    "肉眼": [0, 0, 0, 0, 0, 0, 1],
    "\\": [0, 0, 0, 0, 0, 0, 0]
}
def ccd_Hematuria(hematuria_str):
    # 确保输入为字符串类型，进行类型转换
    hematuria_str = str(hematuria_str).strip()
    if hematuria_str in hematuria_mapping:
        return torch.FloatTensor(hematuria_mapping[hematuria_str])
    else:
        raise NotImplementedError(f"Invalid hematuria input: {hematuria_str}")

# 蛋白尿的映射字典
proteinuria_mapping = {
    "0": [1, 0, 0, 0, 0, 0],
    "+-": [0, 1, 0, 0, 0, 0],
    "1+": [0, 0, 1, 0, 0, 0],
    "2+": [0, 0, 0, 1, 0, 0],
    "3+": [0, 0, 0, 0, 1, 0],
    "4+": [0, 0, 0, 0, 1, 0],
    "\\": [0, 0, 0, 0, 0, 0]
}
def ccd_Proteinuria(proteinuria_str):

    proteinuria_str = str(proteinuria_str).strip()
    if proteinuria_str in proteinuria_mapping:
        return torch.FloatTensor(proteinuria_mapping[proteinuria_str])
    else:
        raise NotImplementedError(f"Invalid proteinuria input: {proteinuria_str}")

def convert_label(R):
    """
    参数：
    - R: 复发状态（1=复发，0=未复发/截尾）。
    返回值：
    - 1: 复发。
    - 0: 未复发。
    - -1: 无法判断（缺失或异常）。
    """
    if R == 1:
        return 1
    elif R == 0:
        return 0
    else:
        return -1
