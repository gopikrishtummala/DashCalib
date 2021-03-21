function [ x,y ] = FindIntersection( m1,c1,m2,c2 )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

if m1 ==m2
    x = 0;
    y = 0;
else
    x = (c1-c2)./(m2-m1);
    y = (m1*x)+c1;
end
end

