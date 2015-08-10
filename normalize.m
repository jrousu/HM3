function normalized = normalize( matrix)
  for i = 1 : (size( matrix))(1)
    normalized(i, :) = matrix( i,:) / norm(matrix( i, :));
  end;  