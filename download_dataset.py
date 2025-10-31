import openxlab
openxlab.login(ak='rnkveopam77mwdndl2bq', sk='ko185qjdvojmyl7wpedaadj8dzarng4blxp3dyxe') # 进行登录，输入对应的AK/SK，可在个人中心添加AK/SK

from openxlab.dataset import info
info(dataset_repo='MingyuLiu/moge-dp') #数据集信息查看

# from openxlab.dataset import get
# get(dataset_repo='MingyuLiu/moge-dp', target_path='/path/to/local/folder/') # 数据集下载

from openxlab.dataset import download
download(dataset_repo='MingyuLiu/moge-dp',source_path='/maniskill_mini.zip', target_path='~/code/Franka_RL') #数据集文件下载