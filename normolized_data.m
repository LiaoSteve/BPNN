function [ y] = normolized_data(x,a,b)
    % enter a matrix x ,size(x) is m by 1  
    % b:after normolized max value
    % a:after normolized min value
    y=(x-min(x))*(b-a)/(max(x)-min(x))+a;
end

