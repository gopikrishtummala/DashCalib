clear all
PointsfileHaar = 'Horizantal_DNN30005000.csv';
xli = 1920;
yli = 1080;
MHaar = csvread(PointsfileHaar);
%MDNN = csvread(PointsfileDNN);
M = [];
for i=1:length(MHaar)
x1 = MHaar(i,4);
y1 = MHaar(i,5);
x2 = MHaar(i,6);
y2 = MHaar(i,7);
slope = (y2-y1)/(x2-x1);
intspt = y1 - (slope*x1);
M  = [M; slope, intspt];
plot([y1 x1],[y2 y1],'or')
xlim([0,xli])
ylim([0,yli])
hold on
plot([y1 x1],[y2 y1])
grid on
end
count  =0;
sum  =0;
for i=1:length(M)
if M(i,1)>0
sum = sum+M(i,1);
count = count +1;
end
end
ag = sum/count;    