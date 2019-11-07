% Create Data
% Training data 1~300,Testing data 301~400
data=zeros(400,3); %(x,y,z=f(x,y))
data(:,1:2)=rand(400,2)*(0.7-(-0.8))-0.8; %-0.8~0.7
for i=1:400
    data(i,3)=f(data(i,1),data(i,2));
end
save('data2.mat','data');