# Octave

### matrix   
a = [1,2,3; 3,4,5; 6,7,8]   
b = [1; 5; 7]

a =   

    1   2   3    
    4   5   6    
    7   8   9    

b =    

    1    
    4
    7    

    
**matrix range**    
a[m, n], m is size of raw, n is size of column    
a(1:m,1) first column
a(1, 1:n) first raw   

**calculation**    
a * b   
b .* a(1:3, 1)   each element of b multiply each element of a(1:3, 1)    
b' transpose of b   


**for loop**    

```
for i = 1:m,
    ...
end
```


