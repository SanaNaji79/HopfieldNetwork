%% Neuroscience of Learning Course Project, Fall 1399-00
%Hopfield network
%Sana Aminnaji
%% 
clc 
clear all

% "A"
Input = ...
    ['    OO    ';
     '    OO    ';  ...
     '   OOOO   ';  ...
     '   O  O   ';  ...
     '  OO  OO  ';  ...
     '  O    O  ';  ...
     ' OOOOOOOO ';  ...
     ' OOOOOOOO ';  ...
     'OO      OO';  ...
     'OO      OO'];
 
 % "B"
Input(:, :, 2) =        ...
    ['OOOOOO    ';  ...
     'OOOOOOO   ';  ...
     'OO   OO   ';  ...
     'OOOOOOO   ';  ...
     'OOOOOOO   ';  ...
     'OO   OOO  ';  ...
     'OO    OO  ';  ...
     'OO   OOO  ';  ...
     'OOOOOOO   ';  ...
     'OOOOOO    '];
 
 % "C"
Input(:, :, 3) =        ...
    ['OOOOOOOOOO';  ...
     'OOOOOOOOOO';  ...
     'OO      OO';  ...
     'OO        ';  ...
     'OO        ';  ...
     'OO        ';  ...
     'OO        ';  ...
     'OO      OO';  ...
     'OOOOOOOOOO';  ...
     'OOOOOOOOOO'];
 
 % "H"
Input(:, :, 4) =        ...
    ['OO      OO';  ...
     'OO      OO';  ...
     'OO      OO';  ...
     'OO      OO';  ...
     'OOOOOOOOOO';  ...
     'OOOOOOOOOO';  ...
     'OO      OO';  ...
     'OO      OO';  ...
     'OO      OO';  ...
     'OO      OO'];
 
 % Persian "M"
Input(:, :, 5) =        ...
    ['      OOO ';  ...
     '    OO  OO';  ...
     'OOOOOOOOOO';  ...
     'OOO       ';  ...
     'OO        ';  ...
     'OO        ';  ...
     'OO        ';  ...
     'OO        ';  ...
     'OO        ';  ...
     'OO        '];
 
 
% Use NoisyInput for testing your network ===============================

% Noisy "A"
NoisyInput =        ...
    ['    OO    ';  
     '    OO    ';  ...
     '   OOOO   ';  ...
     '   O  OO  ';  ...
     '  OO  OOO ';  ...
     '  OO   OO ';  ...
     ' OOO  OOO ';  ...
     ' OOOOOOOO ';  ...
     'O       OO';  ...
     'OO      OO'];
 
 % Noisy "B"
NoisyInput(:, :, 2) =        ...
    ['OOOOOOO   ';   ...
     'OOOOOOOOO ';   ...
     'OO   OOOO ';  ...
     'OOOOOOOOO ';   ...
     'OOOOOOO   ';  ...
     'OO   OOO  ';  ...
     'OOOO  OO  ';  ...
     'OO   OOO  ';  ...
     'OOOOOOO   ';  ...
     'OOOOOOOO  '];
 
 % Noisy "C"
NoisyInput(:, :, 3) =        ...
    ['OOOOOOOOOO';  ...
     'OOOOOO OOO';  ...
     'OO    O OO';  ...
     'OO        ';  ...
     'OOOOOO    ';  ...
     'OO    OOO ';  ...
     'OO        ';  ...
     'OO      OO';  ...
     'O  OOOOOOO';  ...
     'OOOOOOOOOO'];
 
 % Noisy "H"
NoisyInput(:, :, 4) =        ...
    ['OO      OO';  ...
     'OO      OO';  ...
     'OO   OO OO';  ...
     'OO      OO';  ...
     'OOO OOOOOO';  ...
     'OOOOOOOOOO';  ...
     'OO      OO';  ...
     'OO OOOO OO';  ...
     'OO      OO';  ...
     'OO    OOOO'];
 
  % Noisy Persian "M"
 NoisyInput(:, :, 5) =        ...
    ['      OOO ';  ...
     '    OO  OO';  ...
     'OO  O  OOO';  ...
     'OOO       ';  ...
     '   OO     ';  ...
     'OO        ';  ...
     'OO     OO ';  ...
     'OO        ';  ...
     'OOOOO     ';  ...
     'OO        '];
%% problem1 
%% example
%%
clc 
clear all

sort1 = perms([1 2 3 4 5]) ;
in1 = [1  1 0 1 1] ;
in2 = [0 0 1 0 1] ;
n = [1 1 1 0 1] ;
a2 = n ;
a1 = [0 0 0 0 0] ;
out = [] ;
w = weights([in1 ; in2]') ;
k = 0 ;
for i = 1:120
    sort2 = sort1(i , :) ;
    a2 = n ;
    a1 = [0 0 0 0 0] ;
    while  ~isequal(a1 , a2)
        a1 = a2 ;
        for j = 1:5
            i = sort2(j) ;
            a2 = cal(a2 , w , i) ;
        end
    end
    out = [out ; a2] ;
    if isequal(a2 , in2)
        k = k + 1 ;
    end
end
%% alphabet
encoded_output = encoding(Input) ;
w1 = weights(encoded_output) ;
w2 = weights1(encoded_output) ;
imagesc(w1) ;
colormap(hot(256)) ;
colorbar;
figure
imagesc(w2) ;
colormap(hot(256)) ;
colorbar;
final1 = Reconstruction(w1 , NoisyInput) ;
final2 = Reconstruction(w2 , NoisyInput) ;
%% problem1
clear
clc
%% loading the images
image1 = imread('C:\Users\Sana\OneDrive\Desktop\semester\neuroscience_karbalaee\homework\HW4_SanaAminnaji_98104722\train\astronaut.png') ;
image2 = imread('C:\Users\Sana\OneDrive\Desktop\semester\neuroscience_karbalaee\homework\HW4_SanaAminnaji_98104722\train\camera.png') ;
image3 = imread('C:\Users\Sana\OneDrive\Desktop\semester\neuroscience_karbalaee\homework\HW4_SanaAminnaji_98104722\train\chelsea.png') ;
image4 = imread('C:\Users\Sana\OneDrive\Desktop\semester\neuroscience_karbalaee\homework\HW4_SanaAminnaji_98104722\train\coffee.png') ;
image5 = imread('C:\Users\Sana\OneDrive\Desktop\semester\neuroscience_karbalaee\homework\HW4_SanaAminnaji_98104722\train\hourse.png') ;
image6 = imread('C:\Users\Sana\OneDrive\Desktop\semester\neuroscience_karbalaee\homework\HW4_SanaAminnaji_98104722\train\motorcycle.png') ;
image = [image1 image2 image3 image4 image5 image6] ;
%% showing the images
for i = 1:6
    grid on ;
    subplot(2 , 3 , i , 'XGrid' , 'on') ;
    imshow(image(: , ((i-1)*128+1):i*128)) ;
    axis([1 128 1 128]) ;
    if i == 2
       title('train images') ; 
    end
    colormap parula
end

%% creating the prototypes
image1 = im2double(image) ;
image_pro1 = zeros(6 , 128*128) ;
for i = 1:6 
    image_pro1(i , :) = reshape(image1(: , ((i-1)*128+1):i*128) , [1 , 128*128]) ;
end
image_pro = 2.*(image_pro1)-1 ;
%% creating the weight matrix
weight = zeros(128*128 , 128*128) ;
for i = 1:128*128
    for j = 1:i
        m = 0 ;
        for t = 1:6
            m = m + image_pro(t , i)*image_pro(t , j) ;
        end
        weight(i , j) = m/(128*128) ;
        weight(j , i) = weight(i , j) ;
    end
end
%% ploting the weight matrix
weight_color = mat2gray(weight) ;
imshow(weight_color) ;
colormap default
colorbar
title('weight for training data') ;
%% checking the initial photos
e = zeros(1 , 6) ;
for i = 1:6
    [a , e(i)] = result(image_pro(i , :) , weight) ;
    subplot(2 , 3 , i) ;
    imshow(a) ;
    if i == 2
        title('checking train images')
    end
    colormap parula
end
%% adding noise to the main images
random = zeros(6 , 3000) ;
for i = 1:6
    random(i , :) = randperm(128*128 , 3000) ;
end
random_image = image_pro ;
im = zeros(6 , 128 , 128) ;
for i = 1:6
    for j = 1:3000
        random_image(i , random(i , j)) = (-1)*image_pro(i , random(i , j)) ;
    end
    output2 = (random_image(i , :) + 1)./2 ;
    output3 = reshape(output2 , [128 , 128]) ;
    im(i , : , :) = output3 ;
    a = mat2gray(output3) ;
    subplot(2 , 3 , i) ;
    imshow(a) ;
    if i == 2
        title('images with added noise') ;
    end
    colormap parula
end
%% computing correlation
for i = 1:6
    x(i , : , :) = corrcoef(random_image(i , :) , image_pro(i , :)) ;
    corr(i) = x(i , 1 , 2) ;
end
%% recreating the images
e1 = zeros(6 , 1) ;
error1 = zeros(6 , 1) ;
for i = 1:6
    [a , e1(i) , t1] = result(random_image(i , :) , weight) ;
    subplot(2 , 3 , i) ;
    imshow(a) ;
    error1(i) = sum((t1-image_pro(i , :)).^2)./(128*128) ;
    if i == 2
        title('recreating the random images')
    end
    colormap parula
end
%% recreating another example
random2 = zeros(6 , 8000) ;
for i = 1:6
    random2(i , :) = randperm(128*128 , 8000) ;
end
random_image2 = image_pro ;
im1 = zeros(6 , 128 , 128) ;
for i = 1:6
    for j = 1:8000
        random_image2(i , random2(i , j)) = (-1)*image_pro(i , random2(i , j)) ;
    end
    output2 = (random_image2(i , :) + 1)./2 ;
    output3 = reshape(output2 , [128 , 128]) ;
    im1(i , : , :) = output3 ;
    a = mat2gray(output3) ;
    subplot(2 , 3 , i) ;
    imshow(a) ;
    if i == 2
        title('images with added noise') ;
    end
    colormap parula
end
figure ;
e1 = zeros(6 , 1) ;
error2 = zeros(6 , 1) ;
for i = 1:6
    [a , e1(i) , t2] = result(random_image2(i , :) , weight) ;
    subplot(2 , 3 , i) ;
    imshow(a) ;
    if i == 2
        title('recreating the random images')
    end
    colormap parula
    error2(i) = sum((t1-image_pro(i , :)).^2)./(128*128) ;
end
%% functions
function [output , error , output1] = result(input , weight) 
output1 = sign(input*weight) ;
output2 = (output1 + 1)./2 ;
output3 = reshape(output2 , [128 , 128]) ;
output = mat2gray(output3) ;
error = sum((output1-input).^2)./(128*128) ;
end

function [output] = encoding(input)
input1 = reshape(input , [size(input , 1)*size(input , 2) , size(input , 3)]) ;
output = zeros(size(input1)) ;
for i = 1:size(input1 , 2)
    for j = 1:size(input1 , 1)
        if input1(j , i) == ' '
            output(j , i) = -1 ;
        else
            output(j , i) = 1 ;
        end
    end
end
end
function [output] = decoding(input , size_neuron)
output1 = zeros(size(input)) ;
for i = 1:size(input , 2)
    for j = 1:size(input , 1)
        if input(j , i) == -1
            output1(j , i) = ' ' ;
        else
            output1(j , i) = 'O' ;
        end
    end
end
output = reshape(output1 , [size_neuron(1) , size_neuron(2) , size(input , 2)]) ;
output = char(output) ;
end
function [weight] = weights(input)
[t , n] = size(input) ;
weight = zeros(t , t) ;
w = zeros(t , t) ;
for i = 1:n
    for j = 1:t
        for k = 1:j
            if j == k
                w(j , k) = 0 ;
            else
                w(j , k) = (2*(input(j , i)) - 1)*(2*(input(k , i)) - 1) ;
                w(k , j) = w(j , k) ;
            end
        end
    end
    weight = weight + w ;
end
end

function [output] = Reconstruction(w , NoisyInput) 
noise = encoding(NoisyInput) ;
%noise = NoisyInput ;
[t , n] = size(noise) ;
out = zeros(size(noise)) ;
for j = 1:n
    a = zeros(t , 1) ;
    b = noise(: , j) ;
    k = 1 ;
    while ~isequal(a , b)|| k==1
        a = b ;
        b = calculate(b , w) ;
        k = k + 1 ;
    end
    out(: , j) = b ;
end
output = decoding(out , [10 ,10]) ;
%output = out ;
end

function [output] = calculate(input , w)
output = input ;
for i = 1:length(input)
    a = sum(output.*w(: , i)) ;
    if a < 0 
        a = -1 ;
    else 
        a = 1 ;
    end
    output(i) = a ;
end
end

function [weight] = weights1(input)
[t , n] = size(input) ;
weight = zeros(t , t) ;
w = zeros(t , t) ;
for i = 1:n
    for j = 1:t
        for k = 1:j
            if j == k
                w(j , k) = 0 ;
            else
                w(j , k) = ((input(j , i)))*((input(k , i))) ;
                w(k , j) = w(j , k) ;
            end
        end
    end
    weight = weight + w ;
end
end
function [output] = cal(input , w , i)
output = input ;
a = sum(input.*w(i , :)) ;
if a < 0 
    a = 0 ;
else 
    a = 1 ;
end
output(i) = a ;
end