import os
import json
import matplotlib.pyplot as plt
import cv2
import csv
import argparse
import glob



# --- Helper Functions (copied and adapted from original script) ---

def parse_count_file(file_path):
    """解析 _count.txt 檔案。"""
    count_dict = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split(': ')
                img_name = parts[0]
                values = parts[1].split(', ')
                tp = int(values[0].split('= ')[1])
                fp = int(values[1].split('= ')[1])
                fn = int(values[2].split('= ')[1])
                count_dict[img_name] = {'tp': tp, 'fp': fp, 'fn': fn}
    except FileNotFoundError:
        print(f"警告：找不到檔案 {file_path}")
        return None
    except Exception as e:
        print(f"錯誤：解析 {file_path} 時發生問題: {e}")
        return None
    return count_dict

def parse_bb_json_file(file_path):
    """解析 _bb.json 檔案。"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"警告：找不到檔案 {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"錯誤：解析 JSON 檔案 {file_path} 時發生問題: {e}")
        return None
    except Exception as e:
        print(f"錯誤：讀取 {file_path} 時發生問題: {e}")
        return None

def get_top10_imgs(count_dict):
    """根據 TP 分數取得前 10 名的圖片。"""
    img_scores = [(img, v['tp']) for img, v in count_dict.items()]
    img_scores.sort(key=lambda x: x[1], reverse=True)
    return [img for img, _ in img_scores[:10]]

def mark_bbox_on_images(count_dict, bb_dict, img_list, dataset_path, output_dir):
    """在圖片上標註 bounding box、標籤和分數並儲存。"""
    os.makedirs(output_dir, exist_ok=True)
    for img_name in img_list:
        if img_name not in bb_dict:
            continue
        
        # Add a check to ensure bb_dict[img_name] is a dictionary with 'tp' and 'fn' keys
        if not isinstance(bb_dict.get(img_name), dict) or 'tp' not in bb_dict[img_name] or 'fn' not in bb_dict[img_name]:
            print(f"警告：在圖片 {img_name} 的資料格式不正確，已跳過: {bb_dict.get(img_name)}")
            continue

        image_path = os.path.join(dataset_path, img_name + ".jpg")
        if not os.path.exists(image_path):
            print(f"警告：找不到圖片 {image_path}")
            continue
        image = cv2.imread(image_path)
        if image is None:
            print(f"警告：無法讀取圖片 {image_path}")
            continue
        
        # 綠色 TP
        for tp_bb in bb_dict[img_name]['tp']:
            label, coords, score = None, None, None
            if isinstance(tp_bb, list) and len(tp_bb) >= 6:
                label = tp_bb[0]
                coords = tp_bb[1:5]
                score = tp_bb[5]
            elif isinstance(tp_bb, dict) and 'bbox' in tp_bb and 'pred_label' in tp_bb:
                label = tp_bb['pred_label']
                coords = tp_bb['bbox']
                score = tp_bb.get('score') # Use .get() for safety
            else:
                if not tp_bb: # Silently skip empty entries
                    continue
                print(f"警告：在圖片 {img_name} 中發現格式不正確的 TP bounding box，已跳過: {tp_bb}")
                continue

            text = f"{label}"
            if score is not None:
                text += f": {score:.2f}"
            
            x_min, y_min, x_max, y_max = map(int, map(float, coords))
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 黃色 FN
        for fn_bb in bb_dict[img_name]['fn']:
            label, coords = None, None
            text = ""
            if isinstance(fn_bb, list) and len(fn_bb) >= 5: # FN from list might not have score
                label = fn_bb[0]
                coords = fn_bb[1:5]
                text = f"{label} (FN)"
            elif isinstance(fn_bb, dict) and 'bbox' in fn_bb and 'gt_label' in fn_bb:
                label = fn_bb['gt_label']
                coords = fn_bb['bbox']
                # FN boxes from gt don't have scores, so we'll just label them as FN
                text = f"{label}"
            else:
                if not fn_bb: # Silently skip empty entries
                    continue
                print(f"警告：在圖片 {img_name} 中發現格式不正確的 FN bounding box，已跳過: {fn_bb}")
                continue

            x_min, y_min, x_max, y_max = map(int, map(float, coords))
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 215, 255), 2)
            cv2.putText(image, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 2)

        # 在左下角標記 TP, FP, FN 計數
        counts = count_dict.get(img_name)
        if counts:
            h, _, _ = image.shape
            font_scale = 0.7
            thickness = 2
            
            tp_text = f"True Positive : {counts['tp']}"
            fp_text = f"False Positive: {counts['fp']}"
            fn_text = f"False Negative: {counts['fn']}"
            
            # Green for TP
            cv2.putText(image, tp_text, (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
            # Red for FP
            cv2.putText(image, fp_text, (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
            # Yellow for FN
            cv2.putText(image, fn_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 215, 255), thickness)

        out_path = os.path.join(output_dir, img_name + ".jpg")
        cv2.imwrite(out_path, image)

# --- Mode Functions ---

def run_auto_mode(input_dir, output_dir, dataset_path):
    """
    Auto Mode: 
    - 為每個模型產生 Top-10 標註圖片。
    - 產生比較所有模型分數的長條圖。
    """
    print("執行 Auto Mode...")
    records = []
    model_dirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])

    for model_name in model_dirs:
        print(f"處理模型: {model_name}")
        model_path = os.path.join(input_dir, model_name)
        count_file = next(glob.iglob(os.path.join(model_path, "*_count.txt")), None)
        bb_file = next(glob.iglob(os.path.join(model_path, "*_bb.json")), None)

        if not count_file or not bb_file:
            print(f"警告：模型 {model_name} 缺少 count 或 bb.json 檔案，已跳過。")
            continue

        count_dict = parse_count_file(count_file)
        bb_dict = parse_bb_json_file(bb_file)
        
        if count_dict is None or bb_dict is None:
            continue

        # 記錄分數 (tp, fp, fn)
        total_tp = sum(v['tp'] for v in count_dict.values())
        total_fp = sum(v['fp'] for v in count_dict.values())
        total_fn = sum(v['fn'] for v in count_dict.values())
        records.append((model_name, total_tp, total_fp, total_fn))

        # 產生 Top-10 圖片
        top10_imgs = get_top10_imgs(count_dict)
        model_auto_output_dir = os.path.join(output_dir, model_name, "auto")
        mark_bbox_on_images(count_dict, bb_dict, top10_imgs, dataset_path, model_auto_output_dir)
        print(f"  - Top-10 圖片已儲存至: {model_auto_output_dir}")

    # 產生匯總結果 (長條圖, records.csv)
    if records:
        # 寫入 records.csv (tp, fp, fn)
        csv_path = os.path.join(output_dir, "records.csv")
        with open(csv_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["model_name", "tp", "fp", "fn"])
            for model_name, tp, fp, fn in records:
                writer.writerow([model_name, tp, fp, fn])

        # 畫長條圖 (tp, fp, fn) 並在每根長條圖上標數值
        names = [r[0] for r in records]
        tps = [r[1] for r in records]
        fps = [r[2] for r in records]
        fns = [r[3] for r in records]

        x = range(len(names))
        width = 0.25
        plt.figure(figsize=(12, 7))
        bars_tp = plt.bar([i - width for i in x], tps, width=width, label='TP', color='g')
        bars_fp = plt.bar(x, fps, width=width, label='FP', color='r')
        bars_fn = plt.bar([i + width for i in x], fns, width=width, label='FN', color='gold')
        plt.ylabel("Count")
        plt.title("Model TP/FP/FN Comparison")
        plt.xticks(x, names, rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()

        # 在每根長條圖上標數值
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                plt.annotate(f'{int(height)}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),  # 3 points vertical offset
                             textcoords="offset points",
                             ha='center', va='bottom', fontsize=10)
        autolabel(bars_tp)
        autolabel(bars_fp)
        autolabel(bars_fn)

        save_path = os.path.join(output_dir, "score_bar.png")
        plt.savefig(save_path)
        print(f"分數長條圖已儲存至: {save_path}")
    else:
        print("沒有足夠的資料來產生匯總結果。")

def run_image_mode(input_dir, output_dir, dataset_path, image_name):
    """
    Image Mode:
    - 針對一張指定的圖片，產生所有模型的標註結果。
    """
    print(f"執行 Image Mode，目標圖片: {image_name}")
    img_name_base = os.path.splitext(image_name)[0]
    model_dirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])

    for model_name in model_dirs:
        model_path = os.path.join(input_dir, model_name)
        count_file = next(glob.iglob(os.path.join(model_path, "*_count.txt")), None)
        bb_file = next(glob.iglob(os.path.join(model_path, "*_bb.json")), None)

        if not count_file or not bb_file:
            print(f"警告：模型 {model_name} 缺少 count 或 bb.json 檔案，已跳過。")
            continue
        
        count_dict = parse_count_file(count_file)
        bb_dict = parse_bb_json_file(bb_file)

        if count_dict is None or bb_dict is None or img_name_base not in bb_dict:
            print(f"警告：在模型 {model_name} 的結果中找不到圖片 {img_name_base} 的資料。")
            continue
        
        # 建立輸出資料夾
        model_image_output_dir = os.path.join(output_dir, model_name, "image")
        os.makedirs(model_image_output_dir, exist_ok=True)
        
        # 標註並儲存圖片
        mark_bbox_on_images(count_dict, bb_dict, [img_name_base], dataset_path, model_image_output_dir)
        print(f"  - 模型 {model_name} 的標註圖已儲存至: {model_image_output_dir}")

def run_score_matrix_mode(input_dir, output_dir):
    """
    Score Matrix Mode:
    - 產生一個 CSV 檔案，比較所有模型在所有圖片上的分數。
    """
    print("執行 Score Matrix Mode...")
    model_names = []
    count_dicts = []
    model_dirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])

    for model_name in model_dirs:
        model_path = os.path.join(input_dir, model_name)
        count_file = next(glob.iglob(os.path.join(model_path, "*_count.txt")), None)
        if count_file:
            count_dict = parse_count_file(count_file)
            if count_dict:
                model_names.append(model_name)
                count_dicts.append(count_dict)
    
    if not count_dicts:
        print("沒有找到任何 count 檔案，無法產生分數矩陣。")
        return

    # 取得所有圖片名稱的聯集
    all_imgs = sorted(list(set.union(*(set(d.keys()) for d in count_dicts))))
    
    # 每張圖片分數計算方式： tp_num*100 - fn_num*1 - fp_num*0
    matrix = []

    for img in all_imgs:
        row = [img]
        for d in count_dicts:
            score = d[img]['tp'] * 100 - d[img]['fn'] if img in d else ''
            row.append(score)
        matrix.append(row)
    # 輸出 CSV
    save_path = os.path.join(output_dir, 'score_matrix.csv')
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image'] + model_names)
        writer.writerows(matrix)
    print(f"分數矩陣已儲存至: {save_path}")

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Qualitative analysis script.")
    
    parser.add_argument(
        '--input-dir', 
        type=str, 
        default='./eval_1_draw_bbox',
        help='存放第一階段評估結果的主資料夾路徑。'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='./eval_2_result',
        help='儲存所有分析結果的主資料夾路徑。'
    )
    parser.add_argument(
        '--dataset-path', 
        type=str, 
        help='存放原始 JPG 圖片的資料夾路徑 (在 auto 和 image mode 中為必要)。'
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['auto', 'image', 'score'], 
        required=True,
        help='執行模式。'
    )
    parser.add_argument(
        '--image-name', 
        type=str,
        help='在 image mode 中要處理的圖片檔案名稱 (例如: "munster_000011_000019_leftImg8bit.jpg")。'
    )

    args = parser.parse_args()

    # 建立主輸出資料夾
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode in ['auto', 'image']:
        if not args.dataset_path:
            parser.error(f"--dataset-path 在 {args.mode} mode 中是必須的。")
    
    if args.mode == 'image':
        if not args.image_name:
            parser.error("--image-name 在 image mode 中是必須的。")

    if args.mode == 'auto':
        run_auto_mode(args.input_dir, args.output_dir, args.dataset_path)
    elif args.mode == 'image':
        run_image_mode(args.input_dir, args.output_dir, args.dataset_path, args.image_name)
    elif args.mode == 'score':
        run_score_matrix_mode(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
