class Employee:

    # Example of instance variable
    __id = 0
    __name = ""
    __surname = ""
    __gender = ""
    __city = ""
    __salary = 0

    # constructor
    def __init__(self,id, name,surname,gender, city, salary):
        self.__id = id
        self.__name=name
        self.__surname=surname
        self.__gender=gender
        self.__city = city
        self.__salary=salary

    def fullname(self):
        return 'name: {} - surname: {}'.format(self.__name, self.__surname)