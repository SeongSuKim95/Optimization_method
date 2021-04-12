import argparse

def f(x):
  # f(x) = a*x^4 + b*x^3 + c*x^2 + d*x + e
  # coefficient = [a,b,c,d,e]
  # 다항식
  temp = args.coefficient

  return temp[0] * (x**4) + temp[1] * (x**3) + temp[2] * (x**2) + temp[3] * x + temp[4]

def f_derivative(x):
  # f'(x) = 4ax^3 + 3bx^2 + 2cx + d
  # coefficient = [a,b,c,d,e]
  # 다항식 1차 미분 함수
  temp = args.coefficient

  return 4*temp[0] * (x**3) + 3*temp[1] * (x**2) + 2*temp[2] * x + temp[3]

def newton():
  max_error = args.Accuracy # 이전 단계에서의 x값과 현재 x값의 차이
  iter = 0 # 반복 횟수
  x = args.Initial_point_Newton # Initial point
  flag = 0 # 조건을 만족할 때 loop를 끝내기 위해 사용
  while flag == 0:
    y = f(x)
    gradient = f_derivative(x) # 1차 미분
    old_x = x
    x = x - y / gradient # Newton's method
    iter += 1
    print(f'step : {iter}, x: {x}, y: {f(x)}, error: {abs(x-old_x)}')
    if abs(x - old_x) < max_error :
        flag = 1

  print(f'Solution: x: {x} y: {y}, error : {abs(x-old_x)}')

def secant():
    max_error = args.Accuracy # 이전 단계에서의 x값과 현재 x값의 차이
    x_0 = args.Initial_point_Secant[0] # Initial point 1`
    x_1 = args.Initial_point_Secant[1] # Initial point 2
    iter = 0 # 반복 횟수
    flag = 0 # 조건을 만족할 때 loop를 끝내기 위해 사용
    print(f"Initial point --> {[x_0, x_1]}")
    while flag == 0:
        y = f(x_0)gitgi
        gradient = (f(x_1) - f(x_0))/(x_1 - x_0) # 1차 미분 함수를 사용하지 않고 평균 변화율을 사용.
        x = x_0 - y / gradient # Newton's method 와 동일한 idea
        iter += 1
        print(f' step : {iter} ,x: {x}, y: {f(x)}, error: {abs(x - x_1)}')
        if abs(x - x_1) < max_error:
            flag = 1
        else:
            x_0 = x_1
            x_1 = x
    print(f'Solution: x: {x} y: {y},error : {abs(x - x_1)}')

def golden_search():

    a = args.range[0] # Initial range a
    b = args.range[1] # Initial range b
    if a < b:
        x_small = a
        x_big = b
    else:
        x_small = b
        x_big = a
    # 구간내에서 대소 비교할 두 지점 x1, x2
    x1 = x_small + 0.382 * (x_big - x_small)
    x2 = x_big - 0.382 * (x_big - x_small)
    flag = 0
    iter = 0
    print(f"Initial point --> x_small :{x_small}, x_big : {x_big},Interval : {x_big - x_small}")
    # 반복 수행을 멈출 최소 구간 크기
    confidence_range = args.Accuracy

    while flag == 0:
        iter += 1

        if f(x1) > f(x2):
            x_small = x1
            Interval = abs(x_big - x_small) # Interval 수정
            x1 = x2 # Golden ratio이므로 계산하지 않아도 된다.
            x2 = x_big - 0.382 * (x_big - x_small) # x2는 변경된 interval에 따라 다시 계산
            print(f"Iteration : {iter}, x_small : {x_small}, x_big : {x_big}, x_opt = {(x_small + x_big)/2}, Interval : {Interval}")
        elif f(x1) <= f(x2):
            x_big = x2
            Interval = abs(x_big - x_small) # Interval 수정
            x2 = x1# Golden ratio이므로 계산하지 않아도 된다.
            x1 = x_small + 0.382 * (x_big - x_small) # x1는 변경된 interval에 따라 다시 계산
            print(f"Iteration : {iter}, x_small : {x_small}, x_big : {x_big}, x_opt = {(x_small + x_big)/2}, Interval : {Interval}")

        if Interval < confidence_range or iter ==100 :
            flag = 1

    print(f"Result : x_small : {x_small}, x_big : {x_big}, x_opt = {(x_small + x_big)/2}, Interval : {Interval}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'Optimization')

    parser.add_argument('--method', required = True, type = str) # 어떤 알고리즘을 쓸지
    parser.add_argument('--coefficient', required = True, nargs = 5, type = float) # 다항식 계수
    parser.add_argument('--range', required = False, nargs =2 ,type = float) # Golden section search 구간
    parser.add_argument('--Initial_point_Newton', required = False, type = float) # Newton's method initial point
    parser.add_argument('--Initial_point_Secant', required = False, nargs = 2, type = float) # Secant method initial point 2개
    parser.add_argument('--Accuracy', required = True, type = float) # Iteration을 멈출 기준
    args = parser.parse_args()


    if args.method == "golden" :
        golden_search()
    elif args.method == "newton" :
        newton()
    elif args.method == "secant" :
        secant()
