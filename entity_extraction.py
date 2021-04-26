# -*- coding: utf-8 -*-
 

import pandas as pd
import json
from glob import glob
import errno
import time
import os
import re
from PIL import Image
import os
from glob import glob
import time
import subprocess
import pandas as pd
import re
from pathlib import Path
import Levenshtein
import pdf2image
import csv
import pandas as pd
import numpy as np
import cv2
import pytesseract
import boto3
import imutils
from google.cloud import vision
import io

def pdf2jpg(file_name):
    DPI = 200
    FIRST_PAGE = None
    LAST_PAGE = None
    FORMAT = 'jpg'
    THREAD_COUNT = 1
    USERPWD = None
    USE_CROPBOX = False
    STRICT = False

    pil_images = pdf2image.convert_from_path(file_name, dpi=DPI, first_page=FIRST_PAGE, last_page=LAST_PAGE, fmt=FORMAT, thread_count=THREAD_COUNT, userpw=USERPWD, use_cropbox=USE_CROPBOX, strict=STRICT)
    return pil_images

def convert_to_jpg(file_name,target_file_path):

    if file_name.lower().endswith(('.png',  '.jpeg', '.bmp','.jpg')):
        img = Image.open(file_name)
        #target_file_name = '.'.join(file_name.split('.')[:-1]) + '.jpg'
        target_file_name = '.'.join(target_file_path.split('.')[:-1]) + '.jpg'
        cv2.imwrite(target_file_name, np.asarray(img))
 
    elif file_name.lower().endswith(('.tif', '.tiff')):
        tiff_pages = []
        tiffstack = Image.open(file_name)
        tiffstack.load()
        for i in range(tiffstack.n_frames):
            tiffstack.seek(i)
            #target_file_name = '.'.join(target_file_path.split('.')[:-1]) + '_page_' + str(i) + '.jpg'
            target_file_name = '.'.join(target_file_path.split('.')[:-1]) + '.jpg'
            cv2.imwrite(target_file_name, np.asarray(tiffstack))
            tiff_pages.append(target_file_name)
            break 
 

    elif file_name.lower().endswith(('.pdf')):
        pdf_image = pdf2jpg(file_name)
       
        for i, image in enumerate(pdf_image):
            #target_file_name = '.'.join(target_file_path.split('.')[:-1]) + '_page_' + str(i) + '.jpg'
            target_file_name = '.'.join(target_file_path.split('.')[:-1]) + '.jpg'
            image.save(target_file_name)
            break
 
    else:
        print("Unknown file format: ", file_name)
 
        
        
def convert_all_files(input_images_folder_path,doc_types,processed_images_path):
    start_time = time.time()
    for doc_type in doc_types:        
        img_path = input_images_folder_path+'/'+doc_type+"/*.*"        
        img_names = glob(img_path)
        for filename in img_names:
            target_file_path = processed_images_path+'/'+doc_type+"/"+os.path.basename(filename)
            print(target_file_path)
            try:
                convert_to_jpg(filename,target_file_path) 
            except Exception as e:
                pass
                print("Failed in convert_to_jpg at: " , str(e))
        
    print("Time taken to convert "+ str(len(img_names))+"  files to jpg files: "+str(((time.time() - start_time)/60))+" minutes") 
    
def enhance_image(file_name):

    #resizing the image
    im = Image.open(file_name)
    print("Original Size", im.size)
    
    im2 = im.resize((min(4200, int(im.size[0]*2)), min(4200, int(im.size[1]*2))), Image.BICUBIC)
    
    #Improve the DPI to 1000 & save the enhanced image
    #target_file_name = '.'.join(file_name.split('.')[:-1]) + '_enhance' + '.jpg'
    target_file_name = '.'.join(file_name.split('.')[:-1])  + '.jpg'
    os.remove(file_name)
    im2.save(target_file_name,dpi=(300,300))

    return target_file_name
    
def rotate_image(file_name):

    #Target file
    target_file_name = '.'.join(file_name.split('.')[:-1]) + '_rotated' + '.jpg'

    #Read image
    img = cv2.imread(file_name, 0)
    #print("Image Size: ",img.shape)

    #Flip the foreground
    gray = cv2.bitwise_not(img)
     
    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    #cv2.imwrite('.'.join(file_name.split('.')[:-1]) + '_thresh' + '.jpg', thresh)
    
    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
     
    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)
     
    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle


    print("Angle:", angle)
    if angle != 0.0:
        # rotate the image to deskew it
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(file_name, rotated)
    
    else:
        #rotating by right angle - use pyttesseract to rotate landscape images
        try:
            a=(pytesseract.image_to_osd(file_name,output_type=pytesseract.Output.DICT))
            print("Orientation:", a)
            if(a["orientation"]==270):
                x=ndimage.rotate(img, 270)
                img=x
                print('\nImage Rotated...\n')
            elif(a["orientation"]==90):
                x=ndimage.rotate(img,90)
                img=x
                print('\nImage Rotated...\n')
            elif(a["orientation"]==180):
                x=ndimage.rotate(img,-90)
                img=x
                print('\nImage Rotated...\n')
        except:
            img=img

        cv2.imwrite(file_name, img)
        #print("Image Size: ",img.shape)
    return target_file_name

def remove_grain_noise(file_name):
    target_file_name = '.'.join(file_name.split('.')[:-1]) + '_grain_removed' + '.jpg'
    
    img = cv2.imread(file_name)
    
    kernel = np.ones((5, 5), np.uint8)
    #cv2.dilate(img, kernel, iterations = 1)
    
    kernel = np.ones((5, 5), np.uint8)
    #cv2.erode(img, kernel, iterations = 1)
    
    #bg_img =  cv2.medianBlur(img, 3)
    cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(target_file_name, img)
    
def remove_shadow(file_name):

    #Read image
    img = cv2.imread(file_name)
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
        
#    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)

    target_file_name = '.'.join(file_name.split('.')[:-1]) + '_noshadow' + '.jpg'
    cv2.imwrite(target_file_name, result_norm)
    
    return target_file_name
    
def image_resize(file_name, height):

    #Read image
    image = cv2.imread(file_name, 0)
    print("Image Size: ",image.shape)

    #Resized image
    resized = imutils.resize(image, height=height)

    print("Image resize: ",resized.shape)
    target_file_name = '.'.join(file_name.split('.')[:-1]) + '_resize' + '.jpg'
    cv2.imwrite(target_file_name, resized)

    return target_file_name
    
def image_cleaning(img_path,doc_types):
    
    for doc_type in doc_types:
        img_names = glob(img_path+'/'+doc_type+'/*')
        for filename in img_names:
            '''#enhance_image(filename)
            #remove_flicker(filename)
            remove_grain_noise(filename)
            remove_shadow(filename)
            rotate_image(filename)
            #image_resize(filename, height = 1024) '''
           
            print(filename)
            #inFile_resize = image_resize(filename, height = 1024)
            #print('image_resize')
            inFile_enhance = enhance_image(filename)
            print('enhance_image')
            #inFile_edge = document_edge_detection(inFile_resize)
            #inFile_deflicker = remove_flicker(inFile_edge)
            #inFile_shadow = remove_shadow(inFile_enhance)
            #print('remove_shadow')
            #inFile_rotate = rotate_image(inFile_shadow)
            #print('rotate_image')
            #inFile_enhance = enhance_image(inFile_rotate)
            #print('enhance_image')
            #inFile = image_resize(inFile_enhance, height = 1024) 
           
            
         

def ocr_tesseract(filename):
    img = Image.open(filename)
    text = pytesseract.image_to_string(img)
    return text

def ocr_textract(filename):
    text = ""
    # Read document content
    with open(filename, 'rb') as document:
        imageBytes = bytearray(document.read())

    # Amazon Textract client
    textract = boto3.client('textract')

    # Call Amazon Textract
    response = textract.detect_document_text(Document={'Bytes': imageBytes})

    #print(response)

    # Print detected text
    for item in response["Blocks"]:
        if item["BlockType"] == "LINE":
            text+=item["Text"]+'\n'
            #print (item["Text"])
    return text

def ocr_google(filename):    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="My_First_Project-a532009b184c.json"    
    
    text = None
    # Read document content
    with io.open(filename, 'rb') as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)
    
    #GCP client 
    client = vision.ImageAnnotatorClient()

    # Call Google OCR
    response = client.text_detection(image=image)

    #print(response)

    # Print detected text
    text = response.text_annotations
    #print('Texts:{}'.format(text))
    return text[0].description  

def onlyascii(char):
    if ord(char) < 48 or ord(char) > 127:
        return ''
    else:
        return char  

def convert_jpg_to_text(input_images_folder_path,ocr_files_path,good_words,doc_types,ocr_engine_list):
    start_time = time.time()
    for doc_type in doc_types:
    
        img_path = input_images_folder_path+'/'+doc_type+'/*'
        img_names = glob(img_path)
        file_number=0 
        for filename in img_names:
            try:
                file_number+=1
                print('Processing file:{} {}'.format(file_number,os.path.basename(filename)))
                for ocr_engine in ocr_engine_list:
                    #text_filename = ocr_files_path+'/'+os.path.basename(filename).replace('.tiff','.txt')
                    pre, ext = os.path.splitext(os.path.basename(filename))
                    text_filename = ocr_files_path+'/'+ocr_engine+'/'+doc_type+'/'+pre+'.txt'

                   
                    text_file = open(text_filename, "w+",encoding='utf-8')                     
                    if ocr_engine in 'Tesseract':
                        text = ocr_tesseract(filename)
                    elif ocr_engine in 'Textract':
                        print('using Textract OCR')
                        text = ocr_textract(filename)
                    elif ocr_engine in 'Google':
                        print('using Google OCR')
                        text = ocr_google(filename)                

                    text_file.write("%s" % text)
                    text_file.close()
                   
                    
                    '''
                    #OCR-Correction
                    if ocr_engine in 'Google':                   
                    
                        text = open(text_filename,encoding='utf-8').readlines()
                        text = [x.lower().strip() for x in text]                    
                        ocr_corrector(text,good_words,text_filename)
                    '''    
                        
                    
            except Exception as e:
                pass
                print("Failed in convert_jpg_to_text at: " , str(e))
    print("Time taken to convert "+ str(len(img_names))+" jpg files to text files: "+str(((time.time() - start_time)/60))+" minutes") 







def ocr_corrector(text,good_words,text_filename):
    #os.remove(text_filename)
    text_file = open(text_filename, "w+",encoding='utf-8')
    text_file.truncate()
    for sent in text:
        sentence = sent.split(" ")
        print('Sentence:',sentence)
        new_sent = []
        for word in sentence:
            if not word.lower() in good_words:
                for gword in good_words:
                    if not word.lower() == gword.lower():
                        if Levenshtein.ratio(word,gword) >= 0.75:
                            word = gword.strip()
            #Check if word is in English
            if re.search('[ -~]+', word, re.I):                
                new_sent.append(word)
        #print('New Sentence:',new_sent)
        text_file.write(" ".join(new_sent)+"\n")
    text_file.close()    
    
    

def getInvoiceID(text):
    if 'devi' in text:
        pattern='([a-z0-9]+)\s*\d{1,2}-[a-z]+-\d{4}'
        value = re.search(pattern, text, re.I)
        if value:
            #print(value.group(1))
            return value.group(1).replace(' ','')
        
    if 'GIRIAS' in text:
        pattern='(?:Invoice No|no|INVOICE NO :)\s*(.*\/.*\/[0-9]+)'
        value = re.search(pattern, text, re.I)
        if value:
            #print(value.group(1))
            return value.group(1).replace(':','')    
    if 'PAI' in text:
        #print('PAI')
        #print(text)
        patterns=['(?:Involce No.|Invoice No.)\s*([a-z]+-[0-9]+)','([a-z]+-[0-9]+).*\s*.*\s*(?:Invoice No.)']
        for pattern in patterns:
            value = re.search(pattern, text,  re.I|re.MULTILINE)
            if value:
                #print(value.group(1))
                return value.group(1).replace(' ','')
                
                
def getInvoicedate(text):
    if 'devi' in text:
        pattern='(\d{1,2}-[a-z]+-\d{4})'
        value = re.search(pattern, text, re.I)
        if value:
            #print(value.group(1))
            return value.group(1).replace(' ','')
        
    if 'GIRIAS' in text:
        #print('GIRIAS')
        patterns=['(?:Invoice Date|ate:|TE:|DATE: :|DATE E:|DATI TE1)\s*(\d{1,2}-[a-z]+-\d{4})',
                    '(?:Invoice Date|ate:|TE:|DATE: :|DATE E:|DATI TE1)\s*(\d{1,2}\/[a-z]+\/\d{4})',
                    '(\d{1,2}-[a-z]+-\d{4})','(\d{1,2}\/[a-z]+\/\d{4})']
        for pattern in patterns:
            value = re.search(pattern, text, re.I)
            if value:
                #print(value.group(1))
                return value.group(1).replace(' ','')    
    if 'PAI' in text:
        #print('PAI')
        pattern='(\d{1,2}-\d{1,2}-\d{4})'
        value = re.search(pattern, text, re.I)
        if value:
            #print(value.group(1))
            return value.group(1).replace(' ','')
#(?:TOTAL)\s*([\d]+\.*[\d]+)            
def getTotalAmount(text):
    if 'devi' in text:
        patterns=[  '(?:TOTAL|Total)\s*\d*\s*[A-Z]+\s*[A-Z]*\s*([\d]+,*[\d]+\.*[\d]+)',
                    '(?:TOTAL|Total)\s*\d*\s*[A-Z]*\s*([\d]+,+[\d]+\.*[\d]+)',
                    '(?:TOTAL|Total)\s*([\d]+,*[\d]+\.*[\d]+)',
                    '(?:Total)\s*\d*\s*[a-z]*\s*[A-Z]*\s*([\d]+,*[\d]+\.*[\d]+)']
        
        for pattern in patterns:
            value = re.search(pattern, text, re.I)
            if value:
                #print(value.group(1))
                return value.group(1).replace(' ','')
        
    if 'GIRIAS' in text:
        #print('GIRIAS')
        patterns=['(?:TOTAL|Total)\s*\d*\s*\d*\.*\d*\s*([\d]+,*[\d]+\.*[\d]{1,2})',
                  '(?:TOTAL|Total)\s*\d*\s*\d*\s*\d*\.*\d*\s*([\d]+,*[\d]+\.*[\d]{1,2})',                  
                  '(?:TOTAL|Total)\s*[A-Z]*\s*[A-Z]*\s*\d*\s*\d*\.*\d*\s*([\d]+,*[\d]+\.+[\d]{1,2})',
                  '(?:TOTAL|Total)\s*[A-Z]*\s*\d*\s*\d*\s*\d*\.*\d*\s*([\d]+,*[\d]+\.*[\d]{1,2})',
                  '(?:TOTAL|Total)\s*(.*)']
        for pattern in patterns:
            value = re.search(pattern, text, re.I)
            if value:
                #print(value.group(1))
                return value.group(1).replace(' ','')    
    if 'PAI' in text:
        #print('PAI')
        patterns=['(?:TOTAL)\s*([\d]+\.*[\d]+)','\s*(\d*,*\d*\.*\d{1,2})\s*(?:TOTAL)']
        for pattern in patterns:
        
            
            value = re.search(pattern, text, re.I)
            if value:
                #print(value.group(1))
                return value.group(1).replace(' ','')
 

       

'''def getCompanyName(text,vendor_list):
    for vendor_name in vendor_list.split('\n'):
      #pattern=r'\b({0})\b'.format(vendor_name)
      pattern = vendor_name
      value = re.findall(pattern, text, re.I)
      if value:
          #print(value)
          return vendor_name '''  
def getCompanyName(text,vendor_list):
    for vendor_name in vendor_list.split('\n'):
      if vendor_name in text:
          #print(value)
        return vendor_name 
      else:
        pattern = vendor_name
        value = re.findall(pattern, text, re.I)
        if value:
          #print(value)
          return vendor_name
      
def getGoodsDescription(FileName,output_dir_path):
    garbage_words = ['Description','Descrpton','Cash','Cheque','nan','State','Gmail','INTERNATIONAL','Name','DESCRIPTION',
                                  'Donation','gmail','Block','Bangalore','TOTAL','Value']
    img_path = output_dir_path+"/*.csv"  
    img_names = glob(img_path)
    for file_path in img_names:
        if FileName in file_path:
        
    
            #file_path = output_dir_path+'/'+FileName+'_Table_1.csv'
            if os.path.exists(file_path):
                output_df = pd.read_csv(file_path)
                #print(output_df)
                goods_desc = list()
                for col_name,col_data in output_df.iteritems():
                    
                    desc_found = False
                    for data in col_data.iteritems():
                        #print(data)
                        if any(el for el in ['Description', 'DESCRIPTION','Descrpton'] if el in str(data[1])):
                            desc_found = True
                            #print('desc_found')
                        if desc_found and not any(ele for ele in garbage_words if ele in str(data[1])):
                            e = str(data[1])
                            if len(e.split('Output')[0])>0:
                                e = e.split('Output')[0]
                            if len(e.split('DUE')[0])>0:
                                e = e.split('DUE')[0]
                            if not any(ele for ele in ['CGST','SGST','Total','Tota','.','%'] if ele in e):
                                goods_desc.append(e)
                    if len(goods_desc)!=0:
                        return str(goods_desc)        
                
                
                '''#print(FileName)
                final_goods_desc = list()
                for goods in goods_desc:
                    if not any(ele for ele in garbage_words if ele in str(goods)):
                        final_goods_desc.append(goods)
                #print(final_goods_desc)
                if len(final_goods_desc)!=0:
                    return str(final_goods_desc)'''
    return ""
    
def timeSince(since):
    now= time.time()
    return now-since
    
def find_value_block(key_block, value_map):
    for relationship in key_block['Relationships']:
        if relationship['Type'] == 'VALUE':
            for value_id in relationship['Ids']:
                value_block = value_map[value_id]
    return value_block
def get_text(result, blocks_map):
    text = ''
    if 'Relationships' in result:
        for relationship in result['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    word = blocks_map[child_id]
                    if word['BlockType'] == 'WORD':
                        text += word['Text'] + ' '
                    if word['BlockType'] == 'SELECTION_ELEMENT':
                        if word['SelectionStatus'] == 'SELECTED':
                            text += 'X ' 
    return text


def get_rows_columns_map(table_result, blocks_map):
    rows = {}
    for relationship in table_result['Relationships']:
        if relationship['Type'] == 'CHILD':
            for child_id in relationship['Ids']:
                cell = blocks_map[child_id]
                if cell['BlockType'] == 'CELL':
                    row_index = cell['RowIndex']
                    col_index = cell['ColumnIndex']
                    if row_index not in rows:
                        # create new row
                        rows[row_index] = {}                        
                    # get the text value
                    rows[row_index][col_index] = get_text(cell, blocks_map)
    return rows
    
def get_kv_relationship(key_map, value_map, block_map):    
    kvs = {}
    for block_id, key_block in key_map.items():
        value_block = find_value_block(key_block, value_map)
        key = get_text(key_block, block_map)
        val = get_text(value_block, block_map)
        kvs[key] = val
    return kvs
def generate_table_csv(file_name,table_result, blocks_map, table_index):
    rows = get_rows_columns_map(table_result, blocks_map)
    table_id = 'Table_' + str(table_index)
    f_name=os.path.splitext(file_name)[0]
    rows_df=pd.DataFrame.from_dict(rows,orient='index').reset_index()
    rows_df.to_csv(f_name+'_'+table_id+'.csv')
    
    # get cells.
    csv = 'Table: {0}\n\n'.format(table_id)

    for row_index, cols in rows.items():
        
        for col_index, text in cols.items():
            csv += '{}'.format(text) + ","
        csv += '\n'        
    csv += '\n\n\n'
    return csv
    
    
def get_form_table_csv_results(file_name):
    start=time.time()
    f_name=os.path.splitext(file_name)[0]
    #print(file_name)
    with open(file_name, 'rb') as file:
        img_test = file.read()
        bytes_test = bytearray(img_test)
        print('Image byte data read')

   # get the results
    client = boto3.client('textract')

    response = client.analyze_document(Document={'Bytes': bytes_test},
                                       FeatureTypes=['TABLES','FORMS'])

    # Get the text blocks
    blocks=response['Blocks']
#    print(response)
#    print(type(response))
#    os.mkdir(f_name)
#    os.chdir(os.getcwd()+'\\'+f_name)
    text=""
    #  for block in blocks:
    for item in response["Blocks"]:
        if item["BlockType"] == "LINE":
            text+=item["Text"]+'\n'
    with open(f_name+"_text.txt", "w") as text_file:
        text_file.write(text)
             

   #  storing the JSON
    
    #blocksStr = ' '.join([str(elem) for elem in blocks]) 
    #doubleQString = "{0}".format(blocksStr)
    with open(f_name+"_blocks.json", "w") as f:
        json.dump(response ,f)
        
  
    # get key and value maps
    key_map = {}
    value_map = {}
    block_map = {}
    for block in blocks:
        block_id = block['Id']
        block_map[block_id] = block
        if block['BlockType'] == "KEY_VALUE_SET":
            if 'KEY' in block['EntityTypes']:
                key_map[block_id] = block
            else:
                value_map[block_id] = block
    
    # get table maps
    
    blocks_map = {}
    table_blocks = []
    for block in blocks:
        blocks_map[block['Id']] = block
        if block['BlockType'] == "TABLE":
            table_blocks.append(block)

    csv = ''
    for index, table in enumerate(table_blocks):
        
        csv += generate_table_csv(file_name,table, blocks_map, index +1)
        csv += '\n\n'
    print(timeSince(start))
    
    return csv,key_map, value_map, block_map

def get_amazon_textract_output_specific_classes(filename,path):
 
    #f_name=os.path.splitext(filename)[0]
 
    #create folder for each pdf
    #os.mkdir(f_name)
    FileName = os.path.basename(filename).replace('.jpg','')
    new_path=path+'/'+FileName
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    
    #new_path=os.getcwd()+'\\'+f_name
    
    
    
    shutil.copy(filename, new_path)
    #print(filename)
    
    #print(new_path)
    FileName = os.path.basename(filename).replace('.jpg','')
    #print(FileName)
    #print('folder created,File copied')
    #os.chdir(os.getcwd()+'\\'+f_name)    

    #f_name=os.path.splitext(i)[0]    
   
    csv,key_map, value_map, block_map=get_form_table_csv_results(new_path+'/'+FileName+'.jpg')   
    #csv,key_map, value_map, block_map=get_form_table_csv_results(new_path+'/'+FileName+'.jpg')       
    kvs = get_kv_relationship(key_map, value_map, block_map)
    kvs_df=pd.DataFrame.from_dict(kvs,orient='index').reset_index()
    kvs_df.to_csv(new_path+'/'+FileName+'_key_value.csv')


def process_bulk_folder_aws_textract(path,textract_output_path):
 
    img_path = path+"/*.jpg"    
    img_names = glob(img_path)
    for filename in img_names:
        get_amazon_textract_output_specific_classes(filename,textract_output_path)
    
         
def generate_invoice_output(ocr_files_path,output_dir_path,output_df,ocr_engine,vendor_list):
     
    start_time = time.time()
    img_path = ocr_files_path+"/*.txt"
    print('ocr_files_path:{}'.format(ocr_files_path))
    img_names = glob(img_path)
    for filename in img_names:
        text_file = open(filename, "r+",encoding='utf-8')
        text = text_file.read()
        print(filename)
        #print(text)
        FileName = os.path.basename(filename).replace('.txt','')
        output_df = output_df.append({'FileName' : FileName } , ignore_index=True)        
        output_df.loc[output_df['FileName']==FileName, 'Company Name'] =  getCompanyName(text,vendor_list)  
        output_df.loc[output_df['FileName']==FileName, 'Invoice ID'] = getInvoiceID(text)
        output_df.loc[output_df['FileName']==FileName, 'Invoice date'] = getInvoicedate(text)
        output_df.loc[output_df['FileName']==FileName, 'Goods Description'] = getGoodsDescription(FileName,output_dir_path) 
        output_df.loc[output_df['FileName']==FileName, 'Total'] = getTotalAmount(text)
        
               
                
   # print(output_df)    
    #output_df.to_excel(output_dir_path+'/invoice_output_'+ocr_engine+'.xlsx',index=False)
    output_df.to_excel(output_dir_path+'/invoice_output'+'.xlsx',index=False)
    print("Time taken to generate output from  "+ str(len(img_names))+" ocr files: "+str(((time.time() - start_time)/60))+" minutes")
    return output_df

import shutil
def copy_textract_tables(source_folder,output_dir_path):
     
    for root, dirs, files in os.walk(source_folder):  
       for file in files:
          path_file = os.path.join(root,file)
          if 'Table' in path_file:
            #print(path_file)
            shutil.copy2(path_file,output_dir_path)
            
def copy_textract_text(source_folder,output_dir_path):
     
    for root, dirs, files in os.walk(source_folder):  
       for file in files:
          path_file = os.path.join(root,file)
          if '.txt' in path_file:
            #print(path_file)
            shutil.copy2(path_file,output_dir_path)

if __name__ == '__main__':
    try:
        start_time = time.time()
       
        doc_types = ['Invoice']
        #ocr_engine_list = ['Tesseract','Textract','Azure','Google']
        ocr_engine_list = ['Textract']
        input_path = 'input_files'
        
        output_dir_path = 'output'
        processed_files_path = 'processed_images'
        ocr_files_path = 'OCR'
        good_words = open("good_word.txt",'r').readlines()
        
        vendor_list_file = 'VendorNames.txt'
        vendor_list = open(vendor_list_file, "r+",encoding='utf-8').read()
        
        
        #convert_all_files(input_path,doc_types,processed_files_path)
        
        #image_cleaning(processed_files_path,doc_types)
        #convert_jpg_to_text(processed_files_path,ocr_files_path,good_words,doc_types,ocr_engine_list)    
        
        textract_output_path = 'textract_output'
        #process_bulk_folder_aws_textract(processed_files_path+'/'+'Invoice',textract_output_path)
        #copy_textract_tables('textract_output',output_dir_path)
         
        for ocr_engine in ocr_engine_list:
            
            column_names = ["FileName","Company Name","Invoice ID","Invoice date","Goods Description","Total"]
            invoice_output_df = pd.DataFrame(columns = column_names)
            invoice_ocr_files_path = ocr_files_path+'/'+ocr_engine+'/'+'Invoice'        
            generate_invoice_output(invoice_ocr_files_path,output_dir_path,invoice_output_df,ocr_engine,vendor_list)
        
        
        print("Total time taken for generating final output: "+str(((time.time() - start_time)/60))+" minutes") 
 
         
    except Exception as e:
        print("Failed in script at: " , str(e))
        #exit()