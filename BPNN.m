
clc,clear;
%========================Load Data ======================================
format long;
load('data.mat')
d=normolized_data(data(:,3),0.2,0.8);% map data -->range 0.2~0.8

% Input Data-->1(bias_input),x,y,z(normolized)
x=zeros(400,4);
x(:,1)=1; 
x(:,2:3)=data(1:400,1:2);
x(:,4)=d(1:400);

%======================= Parameters ==============================
iter=200000;            %iteration(cycle)
learning_rate=0.1;
alpha=0.8;                %momentum 
input_cell=3;           % 1(bias_input),x,y
hidden_cell=20;
output_cell=1;          % z=f(x,y)

%hidden layer to input layer:w_h2i
w_h2i=rand(hidden_cell,input_cell)-0.5; %-0.5~0.5
%output layer to hidden layer :w_o2h
w_o2h=rand(output_cell,hidden_cell)-0.5; %-0.5~0.5

%========================= Loss ================================
error=zeros(1,300);
E=zeros(1,300);
E_ave=zeros(1,iter);

%=========================Temp array=================================
yout=zeros(output_cell);
dw_o2h=zeros(output_cell,hidden_cell);
w_o2h_new=zeros(output_cell,hidden_cell);
dw_h2i=zeros(hidden_cell,input_cell );
w_h2i_new=zeros(hidden_cell,input_cell);

%=========================Training=============================
for cycle=1:iter    
    for num=1:300 % #1~#300 data
        %=======================Forward===========================
        % the value of x0*w0+x1*w1+...
        v_hidden=zeros(hidden_cell,1); %hidden layer
        v_output=zeros(output_cell,1); %output layer
        % input layer to hidden layer 
        for j=1:hidden_cell
            for i=1:input_cell
                v_hidden(j,1)=v_hidden(j,1)+x(num,i)*w_h2i(j,i);
            end    
        end
        % hidden layer to output layer
        for j=1:output_cell
            for i=1:hidden_cell
                v_output(j,1)=v_output(j,1)+sigmoid(v_hidden(i,1))*w_o2h(j,i);        
            end      
            yout(j)=sigmoid(v_output(j,1));
        end
        error(1,num)=d(num,1)-yout(j);
        E(num)=1/2*(error(1,num))^2;

        %===================== Back Propagation  ==============================
        for j=1:output_cell
            delta_o2h=error(j,num)*(yout(j)*(1-yout(j)));
            for i=1:hidden_cell            
                dw_o2h(j,i)=alpha*dw_o2h(j,i)+learning_rate*delta_o2h*sigmoid(v_hidden(i,1));
                w_o2h_new(j,i)=w_o2h(j,i)+dw_o2h(j,i);
            end             
        end
        for j=1:hidden_cell     
            delta_h2i=sigmoid(v_hidden(j,1))*(1-sigmoid(v_hidden(j,1)))*delta_o2h*w_o2h(1,j);
            for i=1:input_cell            
                dw_h2i(j,i)=alpha*dw_h2i(j,i)+learning_rate*delta_h2i*sigmoid(x(num,i));
                w_h2i_new(j,i)=w_h2i(j,i)+dw_h2i(j,i);
            end    
        end
        w_o2h=w_o2h_new;
        w_h2i=w_h2i_new;
    end
    E_ave(cycle)=mean(E);    
end

%%
%======================== Predict data  ==========================
% output cell data=1; % z=f(x,y)
predict_data=zeros(100,1);
for num=301:400     
        % the value of x0*w0+x1*w1+...
        v_hidden=zeros(hidden_cell,1); %hidden layer
        v_output=zeros(output_cell,1); %output layer
        % input layer to hidden layer 
        for j=1:hidden_cell
            for i=1:input_cell
                v_hidden(j,1)=v_hidden(j,1)+x(num,i)*w_h2i(j,i);
            end    
        end
        % hidden layer to output layer
        for j=1:output_cell
            for i=1:hidden_cell
                v_output(j,1)=v_output(j,1)+sigmoid(v_hidden(i,1))*w_o2h(j,i);        
            end      
            yout(j)=sigmoid(v_output(j,1));
            predict_data(num-300,1)=yout(j);
        end
        error(1,num)=d(num,1)-yout(j);
        E(num)=1/2*(error(1,num))^2;
end
%%
figure(1)
subplot(211),plot(1:cycle,E_ave),
title(['Training Loss:',num2str(E_ave(cycle)) ,'  Hidden cell :',num2str(hidden_cell),'  \eta:',num2str(learning_rate),'  \alpha:',num2str(alpha)]);
xlabel('cycle'),ylabel('E ave')
subplot(212),stem(301:400,E(301:400),'.'),
title(['Test Loss :',num2str(mean(E(301:400)))]);
xlabel('number of data'),ylabel('E')
suptitle(['E test - E train = ',num2str(mean(E(301:400))-E_ave(cycle))]);

figure(2)
subplot(121)
a=data(301:400,1);
b=data(301:400,2);
c=normolized_data(predict_data(1:100,1),min(data(:,3)),max(data(:,3)));% z--> map to original range
scatter3(a,b,c,'.','r');
xlabel('x'),ylabel('y'),zlabel('z'),title('100 predict data');

subplot(122)
a=data(301:400,1);
b=data(301:400,2);
c=data(301:400,3);
scatter3(a,b,c,'.','b');
xlabel('x'),ylabel('y'),zlabel('z');
title('100 testing true data')


figure(3)
a=data(1:300,1);
b=data(1:300,2);
c=data(1:300,3);
scatter3(a,b,c,'.','b')
xlabel('x'),ylabel('y'),zlabel('z'),title('300 training data')


% Reference--> http://darren1231.pixnet.net/blog/post/338810666-%E9%A1%9E%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF%28backpropagation%29-%E7%AD%86%E8%A8%98