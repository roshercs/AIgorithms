$(".mandatory").on('change',function()
{
    if(seleccionados = $('input:checkbox:checked').length >3)
    {
      alert('No se pueden seleccionar mas de 3 opciones');
      $(this).prop('checked',false);
      return;
    }
});