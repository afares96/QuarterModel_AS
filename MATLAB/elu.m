function result = elu(a)
result = zeros(length(a),1);
for i = 1:length(a)
if a(i) >= 0
    result(i) = a(i);
else 
    result(i) = exp(a(i))-1;

end
end