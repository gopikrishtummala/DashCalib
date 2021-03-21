function [ y ] = DeriveLines( slope, intersept )
x = -1000:0.1:2000;
y = (slope*x) + intersept;
end

