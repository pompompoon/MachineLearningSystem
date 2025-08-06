import os
import shutil
import argparse
import datetime

def organize_results(data_file_path, current_output_dir):
    """
    データファイル名に基づいて結果ファイルを整理するための関数
    
    Parameters:
    -----------
    data_file_path : str
        データファイルへのパス
    current_output_dir : str
        現在の出力ディレクトリ
    """
    # データファイル名から拡張子を除去したベース名を取得
    data_file_name = os.path.basename(data_file_path)
    base_name = os.path.splitext(data_file_name)[0]
    
    # 新しい出力ディレクトリを作成
    # base_nameと現在の日時を含む
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_output_dir = f"{base_name}_{timestamp}"
    
    if not os.path.exists(new_output_dir):
        os.makedirs(new_output_dir)
        print(f"新しい出力ディレクトリを作成しました: {new_output_dir}")
    
    # 現在の出力ディレクトリからすべてのファイルを新しいディレクトリにコピー
    if os.path.exists(current_output_dir):
        file_count = 0
        for file_name in os.listdir(current_output_dir):
            source_path = os.path.join(current_output_dir, file_name)
            if os.path.isfile(source_path):
                destination_path = os.path.join(new_output_dir, file_name)
                shutil.copy2(source_path, destination_path)
                file_count += 1
        
        print(f"{file_count}個のファイルを{current_output_dir}から{new_output_dir}にコピーしました")
    else:
        print(f"警告: 出力ディレクトリ {current_output_dir} が見つかりません")
    
    return new_output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='実験結果ファイルを整理するスクリプト')
    parser.add_argument('--data-path', type=str, required=True,
                       help='データファイルへのパス')
    parser.add_argument('--output-dir', type=str, default="result",
                       help='現在の出力ディレクトリ')
    
    args = parser.parse_args()
    
    new_dir = organize_results(args.data_path, args.output_dir)
    print(f"整理が完了しました。すべての結果は {new_dir} に保存されています。")