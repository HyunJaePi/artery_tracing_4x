'''
class AteryTracing4x 
--- class for artery tracing (4x data)

author: Jay Pi, 2020jaypi@gmail.com
'''

import numpy as np
import tifffile as tiff
from datetime import datetime as dt
from pathlib import Path
import cv2, cc3d, sys 
from glob import glob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd
import numpy as np
import SimpleITK as sitk
from scipy.stats import mode
sys.path.append('./tubemap/')
from tubemap import Skeletonization as skl

       

########### class AteryTracing4x ##########
class ArteryTracing4x:
    def __init__(self, sess):
        self.path_ref = './ref_brains_half/'
        self.path_sess = sess 
        self.set_paths(self.path_sess)
        
        self.dim_z, self.dim_x, self.dim_y = self.get_tif_dim_zxy()
        self.list_iz = self.get_list_iz()
        self.filling_thr1, self.filling_thr2 =  0, 50000   
        self.res_orig = np.array([2, 1.8, 1.8]) # 4x resolution; zxy; z=2 for vessels & 5 for cells
        self.res_dest = np.array([20, 20, 20]) # 20 um
        self.n_cpu = int(cpu_count())
        
    def set_paths(self, path_):
        self.path_intermediate = path_+'intermediate/'
        Path(self.path_intermediate).mkdir(parents=True, exist_ok=True)
        self.path_elastix = self.path_intermediate + 'elastix/'
        Path(self.path_elastix).mkdir(parents=True, exist_ok=True)
        self.path_raw = path_
        Path(self.path_raw+f'stitched_00/').mkdir(parents=True, exist_ok=True)
        self.path_binary = path_+ 'binary/'
        Path(self.path_binary).mkdir(parents=True, exist_ok=True)
        self.path_outputs = path_+'outputs/'
        Path(self.path_outputs).mkdir(parents=True, exist_ok=True)

    def get_tif_dim_zxy(self, ch=0):
        if len(glob(self.path_sess+f'stitched_{ch:02}/Z*_ch{ch:02}.tif'))==0:
            return np.nan, np.nan, np.nan
        else: 
            dim_z = len(glob(self.path_raw+f'stitched_{ch:02}/Z*_ch{ch:02}.tif'))
            fname_tif = glob(self.path_raw+f'stitched_{ch:02}/Z*_ch{ch:02}.tif')[0]
            with tiff.TiffFile(fname_tif) as tif:
                tif_tags = {}
                for tag in tif.pages[0].tags.values():
                    name, value = tag.name, tag.value
                    tif_tags[name] = value
                img = tif.pages[0].asarray()
            return int(dim_z), img.shape[0], img.shape[1]
        
    def get_list_iz(self, ch=0):
        '''get a list of z indices
            input(s): ch -- channel to process
            output(s): a list of z indices
        '''
        list_iz = glob(self.path_raw+f'stitched_{ch:02}/*.tif')
        list_iz = [int(f.split('_')[-2][-5:]) for f in list_iz]
        list_iz.sort()
        return list_iz



### ----- methods -----
### 0. major procecures
    def _00_registration(self):
        self.downsample_parallel(self.res_orig, self.res_dest, self.path_raw+f'stitched_00/', isBinary=False) 
        self.prep_rotated_img()
        self.apply_elastix()
        self.apply_transformix()

    def _01_binarization(self, ch=0):
        # 0) binarize 4x data in parallel
        list_bin = []
        with Pool(processes=self.n_cpu) as pool:
            list_bin.append(pool.map(self.binarization_parallel, ((iz, ch) for iz in self.list_iz)))
        # 1) down-sample binary images
        self.downsample_parallel(self.res_orig, self.res_dest, self.path_binary+f'ch{ch:02}/', isBinary=True) # binary
        # 2) orient img to the coronal
        f_in = self.path_intermediate+f'binary_resampled_ch{ch:02}_20um.tif'
        f_out = self.path_intermediate+f'binary_resampled_ch{ch:02}_rotated.tif'
        self.horizontal2coronal(f_in, f_out)

    def _02_skeletonization(self, ch=0, iter_erosion=7, thr_dust=3):
        # 0) skeletonize binary_ch00 (artery)  
        binary = tiff.imread(self.path_intermediate+f'binary_resampled_ch{ch:02}_rotated.tif')            
        skele_ = skl.skeletonize(binary.astype(bool), sink=None, verbose=False, delete_border=True);
        tiff.imsave(self.path_intermediate+f'skeleton_ch{ch:02}.tif', skele_.astype(bool))
        # 1) clean up
        fin_skele = self.path_intermediate + 'skeleton_ch00.tif'
        fout_skele = self.path_intermediate + f'skele_cl{iter_erosion}.tif'
        path_transf_anno = self.path_intermediate + 'elastix/'
        skele_cl = self.rm_srfc_skele_n_dust(fin_skele, fout_skele, path_transf_anno, iter_erosion, thr_dust)
        # 2) for checking & degugging
        skele_cl_proj = skele_cl.any(axis=0)
        tiff.imsave(self.path_intermediate+f'skele_cl_proj_thr{thr_dust}vxl.tif', skele_cl_proj)

    def _03_generate_output_df(self, fin_skele='skele_cl7.tif'):
        self.quant_skele_ROI(fin_skele)
        df_out = self.get_formated_output_df()

        



### 1. image processing
    def binarization_parallel(self, args_):
        iz, ch = args_
        return self.binarization_single_img(iz, ch)

    def binarization_single_img(self, iz, ch, thr_bg=200, thr_ceil=5000, thr_q=.99, thr_area=10, iter=2, debugging=False): 
        '''artery binarization'''
        fname = self.path_raw+f'stitched_{ch:02}/Z{iz:05}_ch{ch:02}.tif' 
        img = tiff.imread(fname)
        # 0. rm bg noise
        img[img<thr_bg] = 0
        if img.sum()>0: # only process images w/ signals
            # 1. clip, blurring & quantile threshoding
            img = img.clip(0, thr_ceil) 
            img = cv2.GaussianBlur(img, (5,5), 1)# gaussian filter
            # quantile thresholding
            tmp = np.ravel(img)
            thr_ = np.quantile(tmp[tmp>0], thr_q)
            img[img<thr_] = 0
            if debugging: tiff.imsave(self.path_aux+'q99_thr.tif', img)
            # 2. remove blobs, dilation, fill holes & erosion
            img = self.remove_blobs(img.astype(np.uint8), thr_area)
            kernel = np.ones((3,3))
            img = cv2.dilate(img.astype(np.uint8), kernel, iterations=iter)
            img = self.fill_holes(img)
            img = cv2.erode(img, kernel, iterations=iter)
            if debugging: tiff.imsave(self.path_aux+f'eroded.tif', img)
        # 8. save
        path_save = self.path_binary+f'ch{ch:02}/'
        Path(path_save).mkdir(parents=True, exist_ok=True)
        if ~debugging: tiff.imsave(path_save+f'Z{iz:05}_ch{ch:02}.tif', img)
        if debugging: tiff.imsave(self.path_aux+f'Z{iz:05}_ch{ch:02}_bin.tif', img)
        return img

    def remove_blobs(self, img, thr_area):
        compo_num, labeled, compo_stats, componentCentroids = \
        cv2.connectedComponentsWithStats(img.astype(np.uint8), connectivity=8) # perform an area filter on the binary blobs:
        compo_labels = [i for i in range(1, compo_num) if compo_stats[i][4] >= thr_area] # exclude i=0 -- bg      
        return np.where(np.isin(labeled, compo_labels) == True, 255, 0)

    def fill_holes(self, img):
        img_8bit = img.astype(np.uint8) #np.uint8(img * 255)
        contours, hierarchy = cv2.findContours(img_8bit, 1, 2)
        # _, contours, hierarchy = cv2.findContours(img_8bit, 1, 2)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if (area>=self.filling_thr1) & (area<self.filling_thr2):
                img_8bit = cv2.fillPoly(img_8bit, pts =[cnt], color=(255,255,255))    
        return img_8bit

    def downsample_parallel(self, res_from, res_to, path_in, isBinary=False, thr_=700):
        ''' down-sample images using parallel processing -- zxy'''
        print(path_in, res_from, res_to)
        res_from, res_to = np.array(res_from), np.array(res_to)
        flist = glob(path_in+f'*.tif')
        flist.sort()
        scaling_factor = np.round(res_to/res_from).astype(int)
        with Pool(processes=self.n_cpu) as pool:
            list_img = pool.map(self.downsample_img, ((flist[iz], scaling_factor) for iz in np.arange(0, len(flist), scaling_factor[0])))
        img_stack = np.stack(list_img).astype(bool) if isBinary else np.stack(list_img).astype(np.uint16)
        if isBinary!=True: img_stack[img_stack<thr_] = 0 # remove  blurred light in the background
        fname = 'binary_resampled_'+path_in.split('/')[-2] if isBinary else 'resampled_ch'+path_in.split('/')[-2][-2:]
        tiff.imsave(self.path_intermediate+f'{fname}_{res_to[0]}um.tif', img_stack)

    def downsample_img(self, args_):
        f_img, scaling_factor = args_
        img = tiff.imread(f_img) if isinstance(f_img,str) else f_img
        # zxy
        img = img[::scaling_factor[1], ::scaling_factor[2]]
        img[0, :] = 0
        img[-1, :] = 0
        img[:, 0] = 0
        img[:, -1] = 0
        return img
    
    def horizontal2coronal(self, f_in, f_out):
        '''convert horizontal sections to coronal'''
        img0 = tiff.imread(f_in)
        img = np.moveaxis(img0, 0, 1)
        img = np.flip(img, 0)
        img = np.flip(img, 1)
        tiff.imsave(f_out, img)

    def rm_srfc_skele_n_dust(self, skele_fin, skele_fout, path_transf_anno, iter_, thr):
        # 1) dusting first
        skele = tiff.imread(skele_fin)
        skele = self.rm_noise(skele, thr)
        # 2) remove surface 
        mask_ = self.mask_srfc(path_transf_anno, iter_)
        skele[mask_==0] = 0
        tiff.imsave(skele_fout, skele.astype(bool))
        return skele

    def mask_srfc(self, path_, iter_, kernel=np.ones((3,3))):
        '''using tranformixed annotation, create a mask for surface removal'''
        anno = tiff.imread(path_+'transformixed.tif')
        anno[anno>0] = 1
        anno = anno.astype(np.uint8)
        mask_ = np.zeros(anno.shape) 
        for iz in range(anno.shape[0]):
            img_ = anno[iz,:,:]
            mask_[iz,:,:] = cv2.erode(img_, kernel, iterations=iter_)
        return mask_

    def rm_noise(sefl, arr_in, thr):
        '''cc3d dust operation -- remove components smaller than 5 voxels =  100 um (at 20um res)'''
        return cc3d.dust(arr_in, threshold=thr, connectivity=26, in_place=False)



### 2. registration
    def prep_rotated_img(self, ch=0):
        '''prepare a rotated img for alignment to ABA'''
        img0 = tiff.imread(self.path_intermediate+f'resampled_ch{ch:02}_20um.tif')
        img = np.moveaxis(img0, 0, 1)
        img = np.flip(img, 0)
        img = np.flip(img, 1)
        img = img.astype(np.int16)
        tiff.imsave(self.path_intermediate+f'resampled_ch{ch:02}_rotated.tif', img)
        
    def apply_elastix(self, ch=0):
        '''apply Elastix'''
        param_elastix_0 = self.path_ref + '001_parameters_Rigid.txt'
        param_elastix_1 = self.path_ref + '002_parameters_BSpline.txt'
        moving = self.path_ref + 'average_template_coronal_20_pa.tif'
        fixed = self.path_intermediate+f'resampled_ch{ch:02}_rotated.tif'
        elastixImageFilter = sitk.ElastixImageFilter()
        parameterMap0 = sitk.ReadParameterFile(param_elastix_0)
        parameterMap1 = sitk.ReadParameterFile(param_elastix_1)
        elastixImageFilter.SetParameterMap(parameterMap0)
        elastixImageFilter.AddParameterMap(parameterMap1)
        elastixImageFilter.SetFixedImage(sitk.ReadImage(fixed))
        elastixImageFilter.SetMovingImage(sitk.ReadImage(moving))
        elastixImageFilter.SetLogToFile(True)
        elastixImageFilter.SetLogFileName("elastix_out.log")
        elastixImageFilter.SetOutputDirectory(self.path_elastix)
        elastixImageFilter.Execute()
        sitk.WriteImage(elastixImageFilter.GetResultImage(), self.path_elastix+'elastixed.tif')
               
    def apply_transformix(self, ch=0):
        '''apply transformix to annotation img'''
        transformixImageFilter = sitk.TransformixImageFilter() 
        param_transf_0 = self.path_elastix+'TransformParameters.0.txt'
        tranf_param_map_0 = sitk.ReadParameterFile(param_transf_0)
        transformixImageFilter.SetTransformParameterMap(tranf_param_map_0)
        param_transf_1 = self.path_elastix+'TransformParameters.1.txt'
        tranf_param_map_1 = sitk.ReadParameterFile(param_transf_1)
        transformixImageFilter.AddTransformParameterMap(tranf_param_map_1)
        anno = self.path_ref + 'allen_20_anno_16bit_pa_for_LH.tif'
        transformix_movingImage = sitk.ReadImage(anno)
        transformixImageFilter.SetMovingImage(transformix_movingImage)
        transformixImageFilter.SetLogToFile(True)
        transformixImageFilter.SetLogFileName("transformix_out.log")
        transformixImageFilter.SetOutputDirectory(self.path_elastix)
        transformixImageFilter.Execute()
        transformix_resultImage = transformixImageFilter.GetResultImage()
        sitk.WriteImage(transformix_resultImage, self.path_elastix+"transformixed.tif")


### 3. Quantification  -----------------------------------
    def get_n_artery_voxel_cnt(self, arr_skele):
        '''return a 1d-array of artery length & the number of arteries'''
        labels_out, N = cc3d.connected_components(arr_skele, return_N=True)
        stats = cc3d.statistics(labels_out)
        return stats['voxel_counts'][1:], N, labels_out, stats 

    def get_anno_vol(self, anno):
        '''calculate a volume of each ROI from the annotation brain'''
        list_anno_ids = np.unique(anno)
        list_anno_ids = [int(id_) for id_ in list_anno_ids if id_>0]
        df = pd.DataFrame()
        for id_ in tqdm(list_anno_ids):
            df.loc[id_, 'ROI_voxel_cnt'] = (anno==id_).sum()
        df = df.reset_index().rename(columns={'index':'ROI_id'})
        df.to_csv(self.path_intermediate+'anno_vol.csv', index=False)

    def quant_skele_ROI(self, fin_skele):
        '''organize qunatification of skele in each ROI'''
        skele_ = tiff.imread(self.path_intermediate+fin_skele)
        anno = tiff.imread(self.path_intermediate+'elastix/transformixed.tif')
        vxl_cnt, N, arr_cc, _ = self.get_n_artery_voxel_cnt(skele_)
        df = pd.DataFrame()
        df['cc_id'] = np.unique(arr_cc)[1:] # skip background--0
        df['voxel_cnt'] = vxl_cnt
        list_ROI_id = []
        for id_ in tqdm(df.cc_id):
            # pick only one ROI if cc_id crosses multiple regions
            list_ = list(np.unique(anno[arr_cc==id_]))  
            mode_, cnt_ = mode(list_)
            list_ROI_id.append(mode_[0])
        df['ROI_id'] =  list_ROI_id
        df.to_csv(self.path_intermediate+f'df_cc.csv', index=False)
        # anno vol
        self.get_anno_vol(anno)
    
    def get_formated_output_df(self):
        # df_base
        df = pd.read_csv(self.path_intermediate+'df_cc.csv').rename(columns={'ROI_id':'id'})
        df_vol = pd.read_csv(self.path_intermediate+'anno_vol.csv', usecols=['ROI_id', 'ROI_voxel_cnt' ]).rename(columns={'ROI_id':'id'})
        ref = pd.read_csv(self.path_ref+'ARA2_annotation_structure_info_v2.csv')
        df = df.merge(ref[['id', 'parent_id']], on='id', how='left').merge(df_vol, on='id', how='left')
        df = df.groupby('parent_id').agg(mean_voxel_cnt=('voxel_cnt','mean'), 
                                        voxel_cnt=('voxel_cnt','sum'),
                                        n_arteries=('cc_id','count'), 
                                        vol_voxel_cnt=('ROI_voxel_cnt','sum')).reset_index()
        df.rename(columns={'parent_id':'id'}, inplace=True)
        df_base = df.merge(ref[['id', 'acronym', 'parent_id', 'parent_acronym']], on='id', how='left')
        # dict_level
        dict_ctx_level1 = {567:'Cerebrum', 688:'Cerebral cortex', 695:'Cortical plate', 315:'Isocortex'}
        dict_ctx_level2 = {184:'FRP', 1057:'GU', 972:'PL', 44:'ILA', 
                        677:'VISC', 541:'TEa', 922:'PERI', 895:'ECT'}
        dict_ctx_level3 = {985:'MOp', 993:'MOs', 
                        378:'SSs', 322:'SSp',  
                        1011:'AUDd',  1002:'AUDp', 1027:'AUDpo', 1018:'AUDv', 
                        39:'ACAd', 48:'ACAv',
                        723:'ORBl', 731:'ORBm', 746:'ORBvl', 738:'ORBv',
                        104:'AId', 111:'AIp', 119:'AIv',
                        879:'RSPd', 886:'RSPv',  
                        385:'VISp', 394:'VISam', 402:'VISal', 409:'VISl',  417:'VISrl', 
                        425:'VISpl', 533:'VISpm', 20079:'VISmma', 20072:'VISmmp', 
                        20065:'VISm', 20086:'VISlla', 20093:'VISrll', 20113:'VISli', 
                        20100:'VISpor', 20120:'VISa'}
        dict_ctx_level4 = { 353:'SSp-n', 329:'SSp-bfd', 337:'SSp-ll', 345:'SSp-m', 
                        369:'SSp-ul', 361:'SSp-tr', 20128:'SSp-un'}
        # level 4
        df_lvl4 = pd.DataFrame()
        df_lvl4['id'] = dict_ctx_level4.keys()
        df_lvl4['acronym'] = dict_ctx_level4.values()
        cols = ['id', 'acronym', 'parent_id', 'parent_acronym', 'mean_voxel_cnt', 'voxel_cnt', 'n_arteries', 'vol_voxel_cnt']
        df_lvl4 = df_lvl4.merge(df_base[cols])
        df_lvl3a = df_lvl4.groupby('parent_id').agg(parent_acronym=('parent_acronym', 'first'),
                                        mean_voxel_cnt=('mean_voxel_cnt', 'mean'),
                                        voxel_cnt=('voxel_cnt','sum'),
                                        n_arteries=('n_arteries', 'sum'),
                                        vol_voxel_cnt=('vol_voxel_cnt', 'sum')
                                        ).reset_index()
        df_lvl3a.rename(columns={'parent_id':'id', 'parent_acronym':'acronym'}, inplace=True)
        df_lvl3a = df_lvl3a.merge(ref[['id', 'parent_id', 'parent_acronym']], on='id', how='left')
        # level 3
        df_lvl3 = pd.DataFrame()
        df_lvl3['id'] = dict_ctx_level3.keys()
        df_lvl3['acronym'] = dict_ctx_level3.values()
        #cols = ['id','parent_id', 'parent_acronym', 'mean_voxel_cnt', 'n_arteries', 'vol_voxel_cnt']
        df_lvl3 = df_lvl3.merge(df_base[cols])
        df_lvl3 = pd.concat([df_lvl3, df_lvl3a], axis=0)
        # level 2
        df_lvl2a = df_lvl3.groupby('parent_id').agg(parent_acronym=('parent_acronym', 'first'),
                                        mean_voxel_cnt=('mean_voxel_cnt', 'mean'),
                                        voxel_cnt=('voxel_cnt','sum'),
                                        n_arteries=('n_arteries', 'sum'),
                                        vol_voxel_cnt=('vol_voxel_cnt', 'sum')
                                        ).reset_index()
        df_lvl2a.rename(columns={'parent_id':'id', 'parent_acronym':'acronym'}, inplace=True)
        df_lvl2a = df_lvl2a.merge(ref[['id', 'parent_id', 'parent_acronym']], on='id', how='left')
        df_lvl2b =  pd.DataFrame()
        df_lvl2b['id'] = dict_ctx_level2.keys()
        df_lvl2b = df_lvl2b.merge(df_base[cols])
        df_lvl2 = pd.concat([df_lvl2a, df_lvl2b], axis=0)
        # level 1
        df_lvl1 = df_lvl2.groupby('parent_id').agg(parent_acronym=('parent_acronym', 'first'),
                                        mean_voxel_cnt=('mean_voxel_cnt', 'mean'),
                                        voxel_cnt=('voxel_cnt','sum'),
                                        n_arteries=('n_arteries', 'sum'),
                                        vol_voxel_cnt=('vol_voxel_cnt', 'sum')
                                        ).reset_index()
        df_lvl1.rename(columns={'parent_id':'id', 'parent_acronym':'acronym'}, inplace=True)
        df_lvl1 = df_lvl1.merge(ref[['id', 'parent_id', 'parent_acronym']], on='id', how='left')
        # level all combined & output
        df_lvl_all = pd.concat([df_lvl1, df_lvl2, df_lvl3, df_lvl4], axis=0).rename(columns={'id':'ROI_id'})
        df_out_template = pd.read_csv(self.path_ref+'counted_3d_cells.csv', usecols=['ROI_id', 'ROI_name', 'ROI_accronym', 'Structure_order'])
        cols = ['ROI_id', 'mean_voxel_cnt', 'voxel_cnt', 'n_arteries', 'vol_voxel_cnt']
        df_out = df_out_template.merge(df_lvl_all[cols], on='ROI_id', how='left')
        df_out['volume (mm3)'] = df_out.vol_voxel_cnt*0.000001
        df_out['voxel density'] = df_out['voxel_cnt']/df_out['volume (mm3)']
        df_out['artery density'] = df_out['n_arteries']/df_out['volume (mm3)']
        df_out.to_csv(self.path_outputs+'result.csv', index=False)
        return df_out
    

    



