from abc import ABC, abstractmethod

class base_regressor(ABC):
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @abstractmethod
    def fit(self, y, x,optimizer) -> None:
        ...









