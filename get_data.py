import pandas_datareader as pdr
from pandas import ExcelWriter
import numpy as np
import os  
import shutil

#stock list
def get_stock_list():
    file_name = "Yahoo_stock_list/ticket_list.txt"
    print ("reading stock list ...")
   
    tk_file = open(file_name,'r')
    temp_buff = []
    for line in tk_file:
        temp_buff.append(line.strip('\n'))
    ticket_list = temp_buff
    return ticket_list


#download_stock_history data    
def download_stock_his_data(ticket_list,path,begdate,enddate):
    if os.path.exists(path) == False:
               os.mkdir(path)
    files= os.listdir(path)  
    for symbol in ticket_list:
        file_name = r''+symbol+'.csv'
        already_has = False
        print(file_name)
        download_time = 0
        for file1 in files: 
            if file1 == file_name:
               already_has = True
               break
        while already_has == False:
           print(r'Downloading '+file_name)
           try:
              download_time += 1
              price = pdr.data.get_data_yahoo(symbol,'1997-09-30','2017-9-30')
              np.savetxt(path+'/'+file_name, price, delimiter=",")
              print(r'Download '+file_name+' success')
              already_has = True
           except:
              print('download error, re-try')
              if download_time > 2:
                 print(r'Download '+file_name+' failed too many times, move to another one')
                 already_has = True

def rm_too_much_nan_file():
   files= os.listdir(path)
   for file_name in files:
      files_full_name = path+r'/'+file_name
      print(r'rm_too_much_nan_file ....')
      infile = open(files_full_name, 'r')
      temp_buffer = []
      nan_count = 0
      for line in infile:
          if line.find('nan') == 1: #count nan lines
             nan_count += 1
      #save usefull file in anther folder
      infile.close()
      if nan_count < 1:
         shutil.copy(files_full_name,'usefull_data/'+file_name)


if __name__ == "__main__":
   
   begin_date = '1997-09-30'
   end_date = '2017-9-30'
   path = "his_data"  

   tk_list = get_stock_list()
   download_stock_his_data(tk_list,path,begin_date,end_date)
   rm_too_much_nan_file()

