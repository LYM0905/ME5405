% Task 1 display original image
figure();
%subplot(2,4,1);
%read the file
fid = fopen("chromo.txt");
% line feed character
lf = newline; 
% carriage return character
cr = char(13); 
% Reads characters from the file, ignoring line feed and carriage return characters. 
% The read characters are stored in a 64x64 matrix.
img = fscanf(fid, [cr lf '%c'],[64,64]);
% close the file handler
fclose(fid);
% transpose since fscanf returns column vectors
img = img'; 
% convert letters A‐V to their corresponding values in 32 gray levels
img(isletter(img))= img(isletter(img)) - 55;
%convert number literals 0‐9 to their corresponding values in 32 gray
img(img >= '0' & img <= '9') = img(img >= '0' &img <= '9') - 48;
%Converts the image matrix to 8-bit unsigned integers
img = uint8(img);
original_img = img;
imshow(original_img, [0 32],'InitialMagnification','fit');
title('chromo');
%hold on


%---------------------------------------------------------------------
%Task 2 Threshold the image and convert it into binary image
figure();
%subplot(2,4,2);
% get the histogram, scan all pixels
intensity_level = zeros(1,32);
sz = size(img);
for b = 1:sz(1)
    for j = 1:sz(2)
        z = img(b,j);
        intensity_level(z+1) = intensity_level(z+1) + 1;
    end
end

% Display the histogram
%bar(intensity_level);
%title('Histogram of image 1');

% set the maximum variance, total pixels and sigma value
maximum_variance = 10^-12;
total_pixels = sum(intensity_level);
sigma = zeros(1,32);
 
% probability levels of pixels
probability_levels = intensity_level / total_pixels;
 
% devide and calculate
for b = 1:16
    p1 = sum(probability_levels(1:b));
    p2 = sum(probability_levels(b+1:end));
    m1 = dot(0:b-1,probability_levels(1:b))   / p1;
    m2 = dot(b:31,probability_levels(b+1:32)) / p2;
    sigma(b) = sqrt(p1*p2*((m1-m2)^2));
 
end 
maximum_variance = max(sigma);
Otsu = find(sigma == maximum_variance)-1;
binary_img = original_img > Otsu ;
imshow(binary_img,'InitialMagnification','fit');  % Otsu method threshold image
title(['Otsu Thresholding at：', num2str(Otsu)]);



%---------------------------------------------------------------------
%Task 3 Determine an one-pixel thin image of the objects
%Global thresholding based on the otsu method
figure();
%subplot(2,4,3);
% get the image size
binary_img = img < Otsu
[m,n] = size(binary_img);             
raw_img = padarray(binary_img, [1 1], 0);  % padding the image
signal = 0;   % define the enable signal
    [obj_row,obj_col] = find(binary_img == 1);
 
    while signal ~= 1 
        tmp_img = raw_img;
 
        % Step 1
        for p = 2:length(obj_row)
            % for b = 2:length(obj_cols)
 
            rand_array = randperm(length(obj_row));
            i = obj_row(rand_array(1));
            j = obj_col(rand_array(1));
 
            if i == 1 || i == m
                continue
            elseif j == 1 || j == n
                continue
            end
 
            % get the pixel, order is: P1,P2,P3...P8,P9,P2
            core_pixel  = [raw_img(i,j)     raw_img(i-1,j)   raw_img(i-1,j+1) ...
                           raw_img(i,j+1)   raw_img(i+1,j+1) raw_img(i+1,j)   ...
                           raw_img(i+1,j-1) raw_img(i,j-1)   raw_img(i-1,j-1) ... 
                           raw_img(i-1,j)];
 
            A_P1 = 0;    % value change times
            B_P1 = 0;    % non zero neighbors
            for k = 2:9
                if core_pixel (k) < core_pixel (k+1)
                    A_P1 = A_P1 + 1;
                end
                if core_pixel (k) == 1
                    B_P1 = B_P1 + 1;
                end
            end
                
            if ((core_pixel(1) == 1)                                    &&...
                   (A_P1 == 1)                                          &&...
                   ((B_P1 >= 2) && (B_P1 <= 6))                         &&...
                   (core_pixel(2) * core_pixel(4) * core_pixel(6) == 0) &&...
                   (core_pixel(4) * core_pixel(6) * core_pixel(8) == 0))
               raw_img(i, j) = 0;
            end
        end
        
        % when previous image is equal to current image, break the loop
        signal = isequal(tmp_img, raw_img);
        if signal      
           break
        end
        
 %---------------------------------------------------------------------------------
        tmp_img = raw_img;
        % Step 2        
        for p = 2:length(obj_row)
            % for b = 2:length(obj_cols)
 
            rand_array=randperm(length(obj_row));
            i = obj_row(rand_array(1));
            j = obj_col(rand_array(1));
 
            if i== 1 || i == m
                continue
            elseif j==1 || j== n
                continue
            end
 
            core_pixel  = [raw_img(i,j)     raw_img(i-1,j)   raw_img(i-1,j+1) ...
                           raw_img(i,j+1)   raw_img(i+1,j+1) raw_img(i+1,j)   ...
                           raw_img(i+1,j-1) raw_img(i,j-1)   raw_img(i-1,j-1) ... 
                           raw_img(i-1,j)];
 
            A_P1 = 0;
            B_P1 = 0;
            for k = 2:9
                if core_pixel (k) < core_pixel (k+1)
                    A_P1 = A_P1 + 1;
                end
                if core_pixel (k) == 1
                    B_P1 = B_P1 + 1;
                end
            end
 
            if ((core_pixel(1) == 1)                                    &&...
                   (A_P1 == 1)                                          &&...
                   ((B_P1 >= 2) && (B_P1 <= 6))                         &&...
                   (core_pixel(2) * core_pixel(4) * core_pixel(8) == 0) &&...
                   (core_pixel(2) * core_pixel(6) * core_pixel(8) == 0))
               raw_img(i, j) = 0;
            end
        end
        
        % when previous image is equal to current image, break the loop
        signal = isequal(tmp_img, raw_img);
        if signal      
           break
        end
    end
 
    img_thinned = [m,n];
    raw_img = 1 - raw_img;
    for i = 2:m+1
        for j = 2:n+1
            img_thinned(i-1, j-1) = raw_img(i,j);
        end
    end


imshow(img_thinned,'InitialMagnification','fit');
title('Thinned image by Zhang Suen Algorithm with Otsu Thresholding');

%%
%---------------------------------------------------------------------------------
%Task 4 Outline
figure();
%subplot(2,4,4);
[m, n] = size(binary_img);
    outline_img = [m,n];
 
    % row operation
    for i = 1:m
        for j = 2:n-1
            if binary_img(i,j) > binary_img(i, j+1)
                outline_img(i,j) = 1;
            end
            if binary_img(i,j) > binary_img(i, j-1)
                outline_img(i, j-1) = 1;
            end
        end
    end
    
    % column operation
    for i = 2:m-1
        for j = 1:n
            if binary_img(i,j) > binary_img(i+1, j)
                outline_img(i,j) = 1;
            end
            if binary_img(i,j) > binary_img(i-1, j)
                outline_img(i-1, j) = 1;
            end
        end
    end
outline_img = imcomplement(outline_img);       
imshow(outline_img,'InitialMagnification','fit');
title('Outlined image with Otsu Thresholding');

%Task 5 label
figure();
%subplot(2,4,5);

[height, width] = size(binary_img);
    out_img = double(binary_img);
    labels = 1;
 
    % first pass
    for i = 1:height
        for j = 1:width
            if binary_img(i,j) > 0  % processed the point
                neighbors = [];     % get the neighborhood, define as: rows, columns, values
                if (i-1 > 0)
                     if (j-1 > 0 && binary_img(i-1,j-1) > 0)
                         neighbors = [neighbors; i-1, j-1, out_img(i-1,j-1)];
                     end
                     if binary_img(i-1,j) > 0
                         neighbors = [neighbors; i-1, j, out_img(i-1,j)];
                     end
                elseif (j-1) > 0 && binary_img(i,j-1) > 0
                    neighbors = [neighbors; i, j-1, out_img(i,j-1)];
                end
 
                if isempty(neighbors)
                    labels = labels + 1;
                    out_img(i,j) = labels;
                else
                    out_img(i,j) = min(neighbors(:,3));
                    % The third column of neighbors is the value of the upper or left point, 
                    % output is the smaller value of the upper and left label
                end
            end
        end
    end
 
    % second pass
    [row, col] = find( out_img ~= 0 ); % point coordinate (row(i), col(i))   
    for i = 1:length(row)
        if row(i)-1 > 0
            up = row(i)-1;
        else
            up = row(i);
        end
 
        if row(i)+1 <= height
            down = row(i)+1;
        else
            down = row(i);
        end
 
        if col(i)-1 > 0
            left = col(i)-1;
        else
            left = col(i);
        end
 
        if col(i)+1 <= width
            right = col(i)+1;
        else
            right = col(i);
        end
 
        % 8 connectivity
        connection = out_img(up:down, left:right);
        
        [r1, c1] = find(connection ~= 0); 
        % in the neighborhood, find coordinates that ~= 0 
        
        if ~isempty(r1)
            connection = connection(:);          % convert to 1 column vector
            connection(connection == 0) = [];    % remove non-zero value
 
            min_label_value = min(connection);   % find the min label value
            connection(connection == min_label_value) = [];    
            % remove the connection that equal to min_label_value
 
            for k = 1:1:length(connection)
                out_img(out_img == connection(k)) = min_label_value;    
                % Change the original value of k in out_img to min_label_value
            end
        end
    end
 
    u_label = unique(out_img);     % get all the unique label values in out_img
    for i = 2:1:length(u_label)
        out_img(out_img == u_label(i)) = i-1;  % reset the label value: 1, 2, 3, 4......
    end
    
label_num = unique(out_img(out_img > 0));
num = numel(label_num);  % numel function gives the pixel numbers

img_rgb = label2rgb(out_img, 'jet', [0 0 0], 'shuffle');
imshow(img_rgb,'InitialMagnification','fit');
title({'Two Pass method with 8 connectivity, using Otsu Thresholding: ';['Label numbers: ',num2str(num)]});

%Task 6 rotate
figure();
%subplot(2,4,6);
I = original_img;
J_30 = imrotate(I,-30,'bilinear','loose');
imshow(J_30, [0 32], 'InitialMagnification','fit');
title('Rotate 30 degree');
%subplot(2,4,7);
figure();
J_60 = imrotate(I,-60,'bilinear','loose');
imshow(J_60, [0 32], 'InitialMagnification','fit');
title('Rotate 60 degree');
%subplot(2,4,8);
figure();
J_90 = imrotate(I,-90,'bilinear','loose');
imshow(J_90, [0 32], 'InitialMagnification','fit');
title('Rotate 90 degree');
