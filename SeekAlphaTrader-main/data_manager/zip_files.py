import zipfile
import os

def zip_files(file_paths, output_zip):
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        for file in file_paths:
            if os.path.isfile(file):
                zipf.write(file, os.path.join(os.path.basename(os.path.dirname(file)), os.path.basename(file)))
                os.remove(file)
            else:
                print(f"文件 {file} 不存在，跳过。")
                
    print(f"文件已成功打包到 {output_zip}")