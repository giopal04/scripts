import torch
import numpy
from typing import Any

class Text:
    def __init__(self, text: Any, name: str = 'variable') -> None:
        self.text = text
        self.name = name
    
    def __format__(self, format_spec: str) :
        match format_spec:
            case 'inspect':
                return self.get_type()
            
            case 'content':
                inspection = self.get_type()
                min_max = self.max_min(show_details=False) + '\n'
                value = f'{self.name} =\n{self.text}\n'

                return inspection + min_max + value
            
            case 'maxmin':
                return self.max_min()
            
            case _:
                raise ValueError(f'Format specifier {format_spec} does not exist')
    
    def get_type(self) -> str:
        if isinstance(self.text, torch.Tensor):
            to_print = (f'{self.name} type = {type(self.text)}\n',
                        f'{self.name} device = {self.text.device}\n',
                        f'{self.name} dtype = {self.text.dtype}\n',
                        f'{self.name} shape = {self.text.shape}')
        
        elif isinstance(self.text, (list, tuple)):
            to_print = (f'{self.name} type = {type(self.text)}\n',
                        f'{self.name} len = {len(self.text)}')
        
        elif isinstance(self.text, dict):
            to_print = (f'{self.name} type = {type(self.text)}\n',
                        f'{self.name} len = {len(self.text)}\n',
                        f'{self.name} keys = {self.text.keys()}')
        
        elif isinstance(self.text, numpy.ndarray):
            to_print = (f'{self.name} type = {type(self.text)}\n',
                        f'{self.name} device = {self.text.device}\n',
                        f'{self.name} dtype = {self.text.dtype}\n',
                        f'{self.name} shape = {self.text.shape}')

        else:
            return f"Didn't expect {type(self.text)}"
        
        message = ''
        for row in to_print:
            message += row

        return message + '\n'

    def max_min(self, show_details: bool = True) -> str:
        if isinstance(self.text, torch.Tensor):
            _max = torch.max(self.text)
            _min = torch.min(self.text)
            _mean = torch.mean(self.text)
        
        elif isinstance(self.text, numpy.ndarray):
            _max = numpy.max(self.text)
            _min = numpy.min(self.text)
            _mean = numpy.mean(self.text)
        
        else:
            return f'Maximum and minimum are not available for {type(self.text)}'
    
        if show_details:
            to_print = (f'{self.name} type = {type(self.text)}\n',
                        f'{self.name} shape = {self.text.shape}\n',
                        f'{self.name} max = {_max},\tmin = {_min},\tmean = {_mean}')
            
            message = ''
            for row in to_print:
                message += row

            return message + '\n'
        
        else:
            return f'{self.name} max = {_max},\tmin = {_min},\tmean = {_mean}'


if __name__ == '__main__':
    tt = torch.rand([2,3,4])
    print(f'{Text(tt, 'tt'):inspect}')
    
    my_tuple = (1,2,3)
    print(f'{Text(my_tuple, 'my_tuple'):inspect}')

    my_list = [1,2,3]
    print(f'{Text(my_list, 'my_list'):inspect}')

    my_dict = {'1':1, '2':2, '3':3}
    print(f'{Text(my_dict, 'my_dict'):inspect}')

    print(f'{Text(my_dict):inspect}')
    print(f'{Text(my_dict, 'my_dict'):content}')
    print(f'{Text(tt, 'tt'):maxmin}')
    print(f'{Text(tt, 'tt'):content}')
