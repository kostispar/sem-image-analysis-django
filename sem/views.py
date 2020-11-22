from django.shortcuts import render, redirect , HttpResponse
from .forms import *
import os
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
import math
from django.core.files.storage import FileSystemStorage
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from functools import partial
import json

posts = [
    {
        'text':"Welcome to the web application of CNT diameter estimation. This tool is using Image Processing techniques to analyze SEM images of Carbon Nanotubes and estimate automatically the distribution of tube diameters.  "
    }
]

def home(request):
    request.session['var'] = 0

    context = {
        'posts': posts,
    }
    return render(request, 'sem/home.html', context)


@csrf_exempt
@login_required
def estimation(request):
    request.session.set_test_cookie()
    keys = []
    request.session['step_1'] = 'no'
    request.session['step_2'] = 'no'
    request.session['step_3'] = 'no'
    request.session['step_4'] = 'no'
    request.session['value'] = 0
    request.session['unit'] = 0
    keys.append(request.session.session_key)
    request.session['user_id'] = request.user.id
    if request.method == 'POST' and request.FILES['myfile']:
        print(request.FILES['myfile'])
        fs = FileSystemStorage()
        if os.path.isdir('media/'+ str(request.user.id)):
            for file in os.listdir('media/'+ str(request.user.id)):
                os.remove(os.path.join('media/'+ str(request.user.id), file))
        request.session['filename'] = fs.save(str(request.user.id) +'/image.png', request.FILES['myfile'])  # saves the file to `media` folder
        request.session['uploaded_file_url'] = fs.url(request.session['filename'])  # gets the url
        print(request.session['uploaded_file_url'])
        return render(request, 'sem/estimation.html', {
            'uploaded_file_url': request.session['uploaded_file_url']
        })
    if os.path.exists('media/'+ str(request.user.id) + '/image.png'):
        if os.path.exists('media/'+ str(request.user.id) + '/config.json'):
            request.session['uploaded_file_url'] = '/media/' + str(request.user.id) + '/image.png'
            return render(request, 'sem/estimation.html',  {'user_id': request.session['user_id'], 'uploaded_file_url': request.session['uploaded_file_url'], 'status': 'yes'})
        else:
            request.session['uploaded_file_url'] = '/media/' + str(request.user.id) + '/image.png'
            request.session['user_id'] = request.user.id
            return render(request, 'sem/estimation.html',  {'user_id': request.session['user_id'], 'uploaded_file_url': request.session['uploaded_file_url']})
    else:
            return render(request, 'sem/estimation.html')


@csrf_exempt
def calibrate(request):
    if request.method == 'POST':
        request.session['start_x'] = request.POST.get("start_x")
        request.session['start_y'] = request.POST.get("start_y")
        request.session['end_x'] = request.POST.get("end_x")
        request.session['end_y'] = request.POST.get("end_y")
        request.session['start_x_d'] = request.POST.get("start_x_d")
        request.session['start_y_d'] = request.POST.get("start_y_d")
        request.session['end_x_d'] = request.POST.get("end_x_d")
        request.session['end_y_d'] = request.POST.get("end_y_d")
        request.session['value'] = request.POST.get("scale")
        request.session['unit'] = request.POST.get("unit")


        if request.session['end_y'] != None:
            request.session['scale'] = math.sqrt(((int(request.session['start_x']) - int(request.session['end_x'])) ** 2) + ((int(request.session['start_y']) - int(request.session['end_y'])) ** 2))
            print(request.session['scale'])
            request.session['step_1'] = 'yes'
            return render(request, 'sem/display_foto.html', {'uploaded_file_url': request.session['uploaded_file_url'],
                                                             'step_1': request.session['step_1'],
                                                             'step_2': request.session['step_2'],
                                                             'step_3': request.session['step_3'],
                                                             'step_4': request.session['step_4'],
                                                             'scale': request.session['value']})
        if request.session['end_y_d'] != None:
            request.session['min_d'] = math.sqrt(((int(request.session['start_x_d']) - int(request.session['end_x_d'])) ** 2) + ((int(request.session['start_y_d']) - int(request.session['end_y_d'])) ** 2))
            print(request.session['min_d'])
            request.session['step_2'] = 'yes'
            return render(request, 'sem/display_foto.html', {'uploaded_file_url': request.session['uploaded_file_url'],
                                                             'step_1': request.session['step_1'],
                                                             'step_2': request.session['step_2'],
                                                             'step_3': request.session['step_3'],
                                                             'step_4': request.session['step_4'],
                                                             'scale': request.session['value']})
        if request.session['value'] != None:
            print(request.session['value'])
            request.session['step_3'] = 'yes'
            return render(request, 'sem/display_foto.html', {'uploaded_file_url': request.session['uploaded_file_url'],
                                                             'step_1': request.session['step_1'],
                                                             'step_2': request.session['step_2'],
                                                             'step_3': request.session['step_3'],
                                                             'step_4': request.session['step_4'],
                                                             'scale': request.session['value']})
        if request.session['unit'] != None:
            print(request.session['unit'])
            request.session['step_4'] = 'yes'
            return render(request, 'sem/display_foto.html', {'uploaded_file_url': request.session['uploaded_file_url'],
                                                             'step_1': request.session['step_1'],
                                                             'step_2': request.session['step_2'],
                                                             'step_3': request.session['step_3'],
                                                             'step_4': request.session['step_4'],
                                                             'scale': request.session['value']})
    return render(request, 'sem/display_foto.html', {'uploaded_file_url': request.session['uploaded_file_url'],
                'step_1': request.session['step_1'],
                                                     'step_2': request.session['step_2'],
                                                     'step_3': request.session['step_3'],
                                                     'step_4': request.session['step_4'],
                'scale': request.session['value']})

def handler(request):
    return render(request, 'sem/estimation.html')


def calculate(request):
    path = request.session['uploaded_file_url']
    path = path.replace("/media","media")
    json_path = ('media/'+ str(request.user.id) + '/config.json')
    with open(json_path) as json_file:
        data = json.load(json_file)
        request.session['opening_1'] = data['opening_1']
        request.session['opening_2'] = data['opening_2']
        request.session['opening_it'] = data['opening_it']
        request.session['closing_1'] = data['closing_1']
        request.session['closing_2'] = data['closing_2']
        request.session['closing_it'] = data['closing_it']
        request.session['median_fil'] = data['median_fil']
    min_distance = request.session['min_d']
    value = request.session['value']
    scale = request.session['scale']
    unit = request.session['unit']
    print("------------------------------------------------")
    print(request.session['scale'],request.session['unit'],request.session['min_d'],request.session['value'])
    im  = cv2.imread(path)
    im = metadata_removal(im)
    im = eight_bit(im)
    gray = grayscale(im)
    equ = histogram_equalization(gray)
    median = median_filtering(equ, int(request.session['median_fil']))
    otsu = otsu_method(median)
    border = add_border(otsu)
    reverse = reversing(border)
    opening = opening_operation(reverse, np.ones((int(request.session['opening_1']), int(request.session['opening_2'])), np.uint8), int(request.session['opening_it']))
    im = closing_operation(opening, np.ones((int(request.session['closing_1']), int(request.session['closing_2'])), np.uint8), int(request.session['closing_it']))

    print('Starting estimations...')
    total_x_0 = np.empty((len(im), len(im[0])))
    total_x_225 = np.empty((len(im), len(im[0])))
    total_x_45 = np.empty((len(im), len(im[0])))
    total_y_0 = np.empty((len(im), len(im[0])))
    total_y_225 = np.empty((len(im), len(im[0])))
    total_y_45 = np.empty((len(im), len(im[0])))
    indexes_0, indexes_225, indexes_45, indexes, indexes_0_y, indexes_45_y, indexes_225_y, indexes_y = [], [], [], [], [], [], [], []
    total = np.zeros((len(im), len(im[0])))
    total_y = np.zeros((len(im), len(im[0])))
    lst_all_diameters, lst_all_diameters_x, lst_all_diameters_y = [], [], []
    for y in range(len(im)):
        for x in range(len(im[y])):
            if im[y][x] == 0:
                # X-SCAN#
                if im[y][x - 1] == 255:
                    # x0 scan
                    start_0_x = [x, y]
                    end_0_x = [x, y]
                    count_0_x = 1
                    for k in range(x + 1, len(im[y])):
                        if im[y][k] == 0:
                            end_0_x = [k, y]
                            count_0_x += 1
                        else:
                            total_x_0[y][x] = count_0_x * float(request.session['value']) / float(request.session['scale'])
                            indexes_0.append([start_0_x, end_0_x])
                            break
                    # x225 scan
                    count_225_x = 1
                    start_225_x = [x, y]
                    end_225_x = [x, y]
                    for k in range(x + 1, len(im[y])):
                        if im[y + count_225_x][k] == 0:
                            end_225_x = [k, y + count_225_x]
                            count_225_x += 1
                        else:
                            total_x_225[y][x] = count_225_x * 1.4141 * float(request.session['value']) / float(request.session['scale'])
                            indexes_225.append([start_225_x, end_225_x])
                            break
                    # x45 scan
                    count_45_x = 1
                    start_45_x = [x, y]
                    end_45_x = [x, y]
                    for k in range(x + 1, len(im[y])):
                        if im[y - count_45_x][k] == 0:
                            end_45_x = [k, y - count_45_x]
                            count_45_x += 1
                        else:
                            total_x_45[y][x] = count_45_x * 1.4141 * float(request.session['value']) / float(request.session['scale'])
                            indexes_45.append([start_45_x, end_45_x])
                            break

                    # find minimum on X
                    if (total_x_225[y][x] <= total_x_0[y][x]) and (total_x_225[y][x] <= total_x_45[y][x]) and (
                            total_x_225[y][x] > min_distance):
                        total[y][x] = total_x_225[y][x]
                        indexes.append([start_225_x, end_225_x])
                        lst_all_diameters_x.append(total_x_225[y][x])
                        lst_all_diameters.append(total_x_225[y][x])
                    elif (total_x_0[y][x] <= total_x_225[y][x]) and (total_x_0[y][x] <= total_x_45[y][x]) and (
                            total_x_0[y][x] > min_distance):
                        indexes.append([start_0_x, end_0_x])
                        total[y][x] = total_x_0[y][x]
                        lst_all_diameters_x.append(total_x_0[y][x])
                        lst_all_diameters.append(total_x_0[y][x])
                    elif (total_x_45[y][x] <= total_x_225[y][x]) and (total_x_45[y][x] <= total_x_0[y][x]) and (
                            total_x_45[y][x] > min_distance):
                        indexes.append([start_45_x, end_45_x])
                        total[y][x] = total_x_45[y][x]
                        lst_all_diameters_x.append(total_x_45[y][x])
                        lst_all_diameters.append(total_x_45[y][x])
                # Y-SCAN#
                if im[y - 1][x] == 255:
                    # y0 scan
                    start_0_y = [x, y]
                    end_0_y = [x, y]
                    count_0_y = 1
                    for k in range(y + 1, len(im)):
                        if im[k][x] == 0:
                            end_0_y = [x, k]
                            count_0_y += 1
                        else:
                            total_y_0[y][x] = count_0_y * float(request.session['value']) / float(request.session['scale'])
                            indexes_0_y.append([start_0_y, end_0_y])
                            break
                    # y225 scan
                    count_225_y = 1
                    start_225_y = [x, y]
                    end_225_y = [x, y]
                    for k in range(y + 1, len(im)):
                        if im[k][x - count_225_y] == 0:
                            end_225_y = [x - count_225_y, k]
                            count_225_y += 1
                        else:
                            total_y_225[y][x] = count_225_y * 1.4141 * float(request.session['value']) / float(request.session['scale'])
                            indexes_225_y.append([start_225_y, end_225_y])
                            break
                    # y45 scan
                    count_45_y = 1
                    start_45_y = [x, y]
                    end_45_y = [x, y]
                    for k in range(y + 1, len(im)):
                        if im[k][x + count_45_y] == 0:
                            end_45_y = [x + count_45_y, k]
                            count_45_y += 1
                        else:
                            total_y_45[y][x] = count_45_y * 1.4141 * float(request.session['value']) / float(request.session['scale'])
                            indexes_45_y.append([start_45_y, end_45_y])
                            break
                    # find minimum on Y
                    if (total_y_225[y][x] <= total_y_0[y][x]) and (total_y_225[y][x] <= total_y_45[y][x]) and (
                            total_y_225[y][x] > min_distance):
                        total_y[y][x] = total_y_225[y][x]
                        indexes_y.append([start_225_y, end_225_y])
                        lst_all_diameters_y.append(total_y_225[y][x])
                        lst_all_diameters.append(total_y_225[y][x])
                    elif (total_y_0[y][x] <= total_y_225[y][x]) and (total_y_0[y][x] <= total_y_45[y][x]) and (
                            total_y_0[y][x] > min_distance):
                        indexes_y.append([start_0_y, end_0_y])
                        total[y][x] = total_y_0[y][x]
                        lst_all_diameters_y.append(total_y_0[y][x])
                        lst_all_diameters.append(total_y_0[y][x])
                    elif (total_y_45[y][x] <= total_y_225[y][x]) and (total_y_45[y][x] <= total_y_0[y][x]) and (
                            total_y_45[y][x] > min_distance):
                        indexes_y.append([start_45_y, end_45_y])
                        total_y[y][x] = total_y_45[y][x]
                        lst_all_diameters_y.append(total_y_45[y][x])
                        lst_all_diameters.append(total_y_45[y][x])
    avg = round(Average(lst_all_diameters),2)
    avg_x =  round(Average(lst_all_diameters_x),2)
    avg_y = Average(lst_all_diameters_y)
    print('The average diameter in x axis is: ', avg_x)
    print('The average diameter in y axis is: ', avg_y)
    print('The average diameter in both axes is: ', avg)
    print('Starting visualizations...')
    # X AXIS visualization#
    plt.imshow(im, cmap='gray')
    for item in indexes:
        x = [item[0][0], item[1][0]]
        y = [item[0][1], item[1][1]]
        plt.plot(x, y)
        plt.xticks([])
        plt.yticks([])
    plt.title("Detections with Image Processing on x-axis\n Average diameter in " + str(request.session['unit']) + " is : " + str(round(avg_x, 2)))
    path1 = path.replace("image", "image1")
    request.session['uploaded_file_url_1'] = path1.replace("media","/media")
    plt.savefig(path1)
    plt.close()
    data = lst_all_diameters
    plt.hist(data,
             bins=25,
             rwidth=1)
    plt.plot([Average(data), Average(data)], [max(data) * 5, 0], label="average")
    percent_formatter = partial(to_percent,
                                n=len(data))
    formatter = FuncFormatter(percent_formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.xticks(range(0, 250, 20))
    plt.xlabel("nm")
    data_np = np.asarray(data)
    plt.plot([np.percentile(data_np, 25), np.percentile(data_np, 25)], [max(data) * 5, 0], label="Q25", c='black')
    plt.plot([np.percentile(data_np, 75), np.percentile(data_np, 75)], [max(data) * 5, 0], label="Q75", c='indigo')
    data_after_lower = [x for x in data if x > np.percentile(data_np, 25)]
    data_final = [x for x in data_after_lower if x < np.percentile(data_np, 75)]
    plt.title("Distribution of diameters \n Average value: " + str(
        round(Average(data), 2)) +  str(request.session['unit']) +"  \n Average value (after Q25/Q75 normalization): " +
              str(round(Average(data_final), 2)))
    path2 = path1.replace("image1", "image2")
    plt.legend()
    plt.savefig(path2)
    request.session['uploaded_file_url_2'] = path2.replace("media","/media")
    return render(request, 'sem/display_foto.html', {'uploaded_file_url': request.session['uploaded_file_url'],'uploaded_file_url_1': request.session['uploaded_file_url_1'], 'uploaded_file_url_2': request.session['uploaded_file_url_2'],'avg': avg, 'unit': request.session['unit'] })


def to_percent(y, position, n):
    s = str(round(100 * y / n, 2))

    if plt.rcParams['text.usetex']:
        return s + r'$\%$'

    return s + '%'

def metadata_removal(im):
    '''
    removes the bottom part with metadata
    :return: new numpy array of image
    '''
    row_to_delete = 0
    for j in range(len(im)):
        if np.all(im[j] == 0):
            row_to_delete = j
            break
    im = np.delete(im, range(row_to_delete,len(im)),axis = 0)
    return(im)


def eight_bit(im):
    '''
    convert to 8 bit
    :param im:
    :return:
    '''
    image = im.astype(np.uint8)
    return(image)


def grayscale(im):
    '''
    convert to grayscale
    :param im:
    :return:
    '''
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return gray


def histogram_equalization(im):
    '''
    histogram equalization of image
    :param im:
    :return:
    '''
    im = cv2.equalizeHist(im)
    return (im)

def median_filtering(im, val):
    '''
    median filtering of image
    :param im:
    :return:
    '''
    median = cv2.medianBlur(im, val)
    return (median)

def otsu_method(im):
    '''
    apply otsu method for binarization
    :param im:
    :return:
    '''
    otsu = cv2.threshold(im,0,255,cv2.THRESH_OTSU)
    otsu_np = otsu[1]
    return (otsu_np)


def add_border(im):
    '''
    add a black frame around the image
    :param im:
    :return:
    '''
    border = cv2.copyMakeBorder(im, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
    return(border)

def reversing(im):
    '''
    reverse black and white pixels
    :param im:
    :return:
    '''
    im = cv2.bitwise_not(im)
    return (im)

def opening_operation(im,kernel, it):
    '''
    opening operation
    :param im:
    :return:
    '''
    opening = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel, iterations=it)
    return(opening)

def closing_operation(im,kernel,it):
    '''
    closing operation
    :param im:
    :param kernel:
    :return:
    '''
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel, iterations=it)
    return(im)


def Average(lst):
    return sum(lst) / len(lst)
